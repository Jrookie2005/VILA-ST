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
- parse_box_from_text / parse_span_from_text / parse_stpair_from_text
- parse_results / format_text / format_conversations
- bbox_post_refine / temporal_iou / iou
- load_json / load_jsonl

Note: These helpers operate on plain strings that contain control tokens like
"<TEMP-INPUT>", "<TEMP-OUTPUT>", "<HEIGHT-INPUT>", etc., and/or numeric values.
They will convert numeric forms into control-token forms, and extract the float
values into a structured "variables" dict (which should be passed to the model).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, Union


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

def format_point_coordinate(text: str)-> List[float] | None:
    '"Knowing that the image has 512x512 pixels, what is the type of anomaly shown at coordinates (359,410)?"'
    pattern = r"\(\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern, text)
    if match:
        a = float(match.group(1))
        a = a / 512.0
        b = float(match.group(2))
        b = b / 512.0
        return [a, b]
    return None

def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def format_coordinate_in_text(text: str, in_out: str) -> str:
    pattern = r"\(\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\s*\)"

    def replace_inside_braces(match: re.Match) -> str:
        a = float(match.group(1))
        b = float(match.group(2))
        a = _clip01(a / 512.0)
        b = _clip01(b / 512.0)
        return (f" (<WIDTH-{in_out}{round(a,3)}><HEIGHT-{in_out}{round(b,3)}>)")

    return re.sub(pattern, replace_inside_braces, text)


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

def get_variables_from_text(text: List[Dict[str, Any]]):
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
    text = text
    if text is None:
        return None, variables_dict
    for key in variables_dict.keys():
        text, lis = seperate_token_number(text, key)
        variables_dict[key].extend(lis)

    variables = {
        "temporal_input_locations": variables_dict["TEMP-INPUT"],
        "temporal_output_locations": variables_dict["TEMP-OUTPUT"],
        "spatial_height_input_locations": variables_dict["HEIGHT-INPUT"],
        "spatial_height_output_locations": variables_dict["HEIGHT-OUTPUT"],
        "spatial_width_input_locations": variables_dict["WIDTH-INPUT"],
        "spatial_width_output_locations": variables_dict["WIDTH-OUTPUT"],
    }
    return text, variables

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


# Additional utility functions from LLaVA-ST

def print_cuda_memory(total_num):
    import pynvml
    pynvml.nvmlInit()
    for gpu_id in range(total_num):
        if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
            print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
            return 0, 0, 0

        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
        total = round(int(meminfo.total) / (1024**3), 2)
        used = round(int(meminfo.used) / (1024**3), 2)
        free = round(int(meminfo.free) / (1024**3), 2)
        print(f"===== cuda {gpu_id} =====")
        print(f"total: {total} GB")
        print(f"used: {used} GB")
        print(f"free: {free} GB")
        print("==========================")


def temporal_iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    _iou = max(min1 - max0, 0) / (max1 - min0)
    return max(0, _iou)


def iou(box1, box2):
    def s(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    intersection = max(0,
                       (min(box1[2], box2[2]) - max(box1[0], box2[0]))) * max(
                           0, (min(box1[3], box2[3]) - max(box1[1], box2[1])))
    intersection = max(0, intersection)
    union = s(box1) + s(box2) - intersection
    return intersection / union if union != 0 else 0


def load_jsonl(ann_path):
    lis = []
    with open(ann_path, "r") as fp:
        for line in fp:
            try:
                lis.append(json.loads(line))
            except Exception as e:
                print(f"Find expception {e}, ignore line.")
                continue
    return lis


def load_json(ann_path):
    with open(ann_path, "r") as fp:
        anns = json.load(fp)
    return anns


def parse_span_from_text(s):
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"
    match = re.search(pattern, s)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return [start_time, end_time]
    else:
        print("No match found.")
        return [0, 0]


def parse_box_from_text(
    text,
    coords_pattern=(r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
                    r"\s*(\d+(?:\.\d+)?)\]")):
    text = text.replace(" ", "")
    # print(text)
    raw_coords = re.findall(coords_pattern, text)
    if not raw_coords or len(raw_coords[0]) != 4:
        print(text)
        return None
    return list(map(float, raw_coords[0]))


def parse_stpair_from_text(
    text,
    pattern=r"(\d+(?:\.\d+)?)\,\:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
    r"\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]",
):
    text = text.replace(" ", "")
    # print(text)
    raw_coords = re.findall(pattern, text)
    dic = {}
    for i in raw_coords:
        dic[float(i[0])] = list(map(float, i[1:]))
    return dic


def parse_float_from_text(s, peer_str="Ends in", post_str=""):
    pattern = peer_str + r"\s*(\d+(?:\.\d+)?)" + post_str
    match = re.search(pattern, s)
    if match:
        end_time = float(match.group(1))
        return end_time
    else:
        print("No match found.")
        return 0


def filter_svg_bboxes_according_to_frame_id(bboxes: Union[dict, list],
                                            frame_id: list[int]):
    if isinstance(bboxes, dict):
        keys = list(bboxes.keys())
        for k in keys:
            if int(k) not in frame_id:
                del bboxes[k]
    return bboxes


def everytype2str(a):
    pass


def bbox_post_refine(bbox, height, width):
    if height >= width:
        x1, y1, x2, y2 = (i * height for i in bbox)
        pad = (height - width) // 2
        x1 -= pad
        x2 -= pad
    else:
        x1, y1, x2, y2 = (i * width for i in bbox)
        pad = (width - height) // 2
        y1 -= pad
        y2 -= pad
    res = [x1 / width, y1 / height, x2 / width, y2 / height]
    return res


def parse_inputs(inputs):
    pattern = r"<([twh])(\d+(?:\.\d+)?)>"

    def replace_inside_braces(match):
        a = match.group(1)
        b = float(match.group(2))
        return str(round(b, 3))
    res = re.sub(pattern, replace_inside_braces, inputs)
    return res


def parse_results(outputs, task):
    if task.lower() in ["rec", "reg"]:
        results = parse_box_from_text(outputs)
    if task.lower() in ["tvg", "tr"]:
        results = parse_span_from_text(outputs)
    if task.lower() in ["stvg", "svg", "elc"]:
        results = parse_stpair_from_text(outputs)
    if task.lower() == "dgc":
        results = []
        lis = outputs.split("]")
        for i in lis:
            box = parse_box_from_text(i+"]")
            if box is not None:
                results.append(box)
    if task.lower() == "dvc":
        pred_lis = outputs.split("{")
        pred_lis = [p for p in pred_lis if '}' in p]
        results = []
        for pred in pred_lis:
            pred = "{"+pred
            caption = pred.split(",")[-1].split(".")[0]
            timestamp = parse_span_from_text(pred)
            if timestamp != [0, 0]:
                results.append({
                    'timestamp': timestamp,
                    'sentence': caption,
                })
    return results


# def format_text(text: str, mode: str):
#     mapp = {"t": "TEMP", "w": "WIDTH", "h": "HEIGHT"}
#     pattern = r"<([twh])(\d+(?:\.\d+)?)>"

#     def replace_inside_braces(match):
#         a = match.group(1)
#         b = float(match.group(2))
#         return f"<{mapp[a]}-{mode}{round(b,3)}>"
#     res = re.sub(pattern, replace_inside_braces, text)
#     return res

def format_text(text: str, mode: str):
    # 尝试转换Box和coordinate
    text = format_box_in_text(text, in_out=mode)
    text = format_coordinate_in_text(text, in_out=mode)
    return text

def format_conversations(conversations):
    for con in conversations:
        if con["value"] is None:
            continue
        text = con["value"]
        mode = "INPUT" if con["from"] == "human" else "OUTPUT"
        text = format_text(text, mode)
        con["value"] = text
    return get_variables(conversations)