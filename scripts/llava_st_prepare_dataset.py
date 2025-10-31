import argparse
import json
import os
from typing import Any, Dict, List

# Reuse the formatting helpers compatible with LLaVA-ST
from llava.utils.lape_format import (
    format_box_in_text,
    format_span_in_text,
    format_float_in_text,
    get_variables,
)


def process_conversations_entry(conv: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a conversation list [{from, value}, ...], apply the LLaVA-ST
    formatting on both human and gpt turns, then extract variables and
    return {conversations, variables}.
    """
    conversations = []
    for turn in conv:
        if turn.get("value") is None:
            conversations.append(turn)
            continue
        in_out = "INPUT" if turn.get("from") == "human" else "OUTPUT"
        text = turn["value"]
        text = format_box_in_text(text, in_out=in_out)
        text = format_span_in_text(text, in_out=in_out)
        text = format_float_in_text(text, in_out=in_out)
        conversations.append({"from": turn["from"], "value": text})

    conversations, variables = get_variables(conversations)
    return {"conversations": conversations, "variables": variables}


def convert_file(src_path: str, dst_path: str, image_key: str = "image", video_key: str = "video"):
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_items: List[Dict[str, Any]] = []
    for item in data:
        # Expect an item contains either image or video path and a conversations field
        media: Dict[str, Any] = {}
        if image_key in item:
            media["image"] = item[image_key]
        if video_key in item:
            media["video"] = item[video_key]

        conv = item.get("conversations")
        if not isinstance(conv, list):
            # try to synthesize from Q/A style
            q = item.get("question")
            a = item.get("answer")
            if q is not None and a is not None:
                conv = [
                    {"from": "human", "value": ("<image>" + str(q)) if "image" in media else str(q)},
                    {"from": "gpt", "value": str(a)},
                ]
            else:
                continue

        pack = process_conversations_entry(conv)
        out_items.append({**media, **pack})

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset in LLaVA-ST conversations+variables format")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--image_key", default="image", help="Image key name in input JSON")
    parser.add_argument("--video_key", default="video", help="Video key name in input JSON")
    args = parser.parse_args()

    convert_file(args.input, args.output, image_key=args.image_key, video_key=args.video_key)


if __name__ == "__main__":
    main()
