"""
LLaVA-ST compatible LAPE formatting utilities.

This module mirrors the key formatting and variable-extraction helpers used
in LLaVA-ST inference pipeline, so we can reuse the exact data conventions
for training and evaluation in VILA-ST.

Functions included:
- format_1d_box / format_2d_box
- format_box_in_text / format_span_in_text / format_float_in_text
- seperate_token_number / get_variables
- replace_and_normalize (optional post-processing)

Note: These helpers operate on plain strings that contain control tokens like
"<TEMP-INPUT>", "<TEMP-OUTPUT>", "<HEIGHT-INPUT>", etc., and/or numeric values.
They will convert numeric forms into control-token forms, and extract the float
values into a structured "variables" dict (which should be passed to the model).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def format_1d_box(text: str) -> Tuple[float, float] | None:
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"
    match = re.search(pattern, text)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return start_time, end_time
    return None


def format_2d_box(text: str) -> List[float] | None:
    pattern = (r"\[\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
               r"\s*(\d+(?:\.\d+)?)\s*\]")
    match = re.search(pattern, text)
    if match:
        a = float(match.group(1))
        b = float(match.group(2))
        c = float(match.group(3))
        d = float(match.group(4))
        return [a, b, c, d]
    return None


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def format_box_in_text(text: str, pad_proc: bool = False, **kwargs) -> str:
    """
    Convert a numeric bbox like [x1,y1,x2,y2] in text to control tokens with in/out suffix.
    Example output: " [<WIDTH-OUTPUT0.123><HEIGHT-OUTPUT0.456><WIDTH-OUTPUT0.789><HEIGHT-OUTPUT0.012>]"
    This matches LLaVA-ST's behavior used before variable extraction.
    """
    box = format_2d_box(text)
    if box is None:
        return text

    in_out = kwargs.get("in_out", "OUTPUT")
    pattern = (r"\[\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
               r"\s*(\d+(?:\.\d+)?)\s*\]")

    def replace_inside_braces(match: re.Match) -> str:
        a = float(match.group(1))
        b = float(match.group(2))
        c = float(match.group(3))
        d = float(match.group(4))
        bpt = [a, b, c, d]

        if pad_proc:
            # Keep identical to LLaVA-ST: bbox_post_refine is only used when original image
            # is letterboxed. We skip here to keep util self-contained.
            pass

        bpt = [round(_clip01(x), 3) for x in bpt]
        return (f" [<WIDTH-{in_out}{bpt[0]}><HEIGHT-{in_out}{bpt[1]}>"
                f"<WIDTH-{in_out}{bpt[2]}><HEIGHT-{in_out}{bpt[3]}>]")

    return re.sub(pattern, replace_inside_braces, text)


def format_span_in_text(text: str, in_out: str) -> str:
    span = format_1d_box(text)
    if span is None:
        return text
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"

    def replace_inside_braces(match: re.Match) -> str:
        s = float(match.group(1))
        e = float(match.group(2))
        if s >= e:
            # keep an error token if needed; original prints a warning
            return "{<error>}"
        return (f" {{<TEMP-{in_out}{round(s,3)}><TEMP-{in_out}{round(e,3)}>}} ")

    return re.sub(pattern, replace_inside_braces, text)


def format_float_in_text(text: str, in_out: str) -> str:
    pattern = r"Starts in (\d+(?:\.\d+)?)"

    def replace_inside_braces(match: re.Match) -> str:
        s = float(match.group(1))
        return f" Starts in <TEMP-{in_out}{round(s,3)}> "

    return re.sub(pattern, replace_inside_braces, text)


def seperate_token_number(text: str, token: str) -> Tuple[str, List[float]]:
    pattern = rf'<{token}(.*?)>'
    matches = re.finditer(pattern, text)
    lis = [(m.group(1), m.start()) for m in matches]
    lis = sorted(lis, key=lambda x: x[-1])
    values: List[float] = []
    for k, _ in lis:
        # remove the numeric part from the token in text (keep pure control token)
        text = re.sub(re.escape(f"<{token}{k}>"), f"<{token}>", text, count=1)
        try:
            values.append(eval(k))  # original uses eval to parse int/float
        except Exception:
            continue
    return text, values


def get_variables(conversations: List[Dict[str, Any]]):
    """
    Extract float values from control tokens in all conversation turns, and
    replace those tokens in text back to pure control tokens without numbers.
    Return the modified conversations and the variables dict.
    """
    variables_dict: Dict[str, List[float]] = {
        "TEMP-INPUT": [],
        "TEMP-OUTPUT": [],
        "HEIGHT-INPUT": [],
        "HEIGHT-OUTPUT": [],
        "WIDTH-INPUT": [],
        "WIDTH-OUTPUT": []
    }
    for con in conversations:
        text = con.get("value")
        if text is None:
            continue
        for key in variables_dict.keys():
            text, lis = seperate_token_number(text, key)
            variables_dict[key].extend(lis)
        con["value"] = text

    variables = {
        "temporal_input_locations": variables_dict["TEMP-INPUT"],
        "temporal_output_locations": variables_dict["TEMP-OUTPUT"],
        "spatial_height_input_locations": variables_dict["HEIGHT-INPUT"],
        "spatial_height_output_locations": variables_dict["HEIGHT-OUTPUT"],
        "spatial_width_input_locations": variables_dict["WIDTH-INPUT"],
        "spatial_width_output_locations": variables_dict["WIDTH-OUTPUT"],
    }
    return conversations, variables


def replace_and_normalize(input_str: str, return_token: bool = False) -> str:
    """
    Replace discrete tokens like <WIDTH-005>/<HEIGHT-099>/<TEMP-050> by their
    normalized numeric value (value/99.0), or return the raw value when
    return_token=True.
    """
    pattern = re.compile(r'(<WIDTH-(\d+)>|<HEIGHT-(\d+)>|<TEMP-(\d+)>)')

    def normalize(match: re.Match) -> str:
        if match.group(2):
            value = int(match.group(2))
        elif match.group(3):
            value = int(match.group(3))
        else:
            value = int(match.group(4))

        if return_token:
            return f"{value},"
        normalized_value = value / 99.0
        return f"{normalized_value:.5f},"

    result_str = re.sub(pattern, normalize, input_str)
    return result_str.replace(",]", "]").replace(",}", "}")
