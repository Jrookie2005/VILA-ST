"""
Quick LAPE forward test for NVILA (LlavaLlamaModel).

Fill in your image path and conversation text. This script builds a minimal
batch with one image and one conversation, forwards through the model, and
prints basic outputs. It supports choosing whether to import from VILA-ST or
LLaVA-ST package tree (both export `llava.*`).

Usage (PowerShell):
  python test_nvila_lape_forward.py \
    --pkg-root VILA-ST \
    --model-path <PATH_TO_PRETRAINED_OR_MERGED_MODEL_DIR> \
    --image <PATH_TO_IMAGE_FILE> \
    --prompt "<image>\nWhat is the center time? Please output <TEMP-OUTPUT> only." \
    --temporal-input 0.25 --temporal-output 0.75

Notes:
  - The prompt must contain one <image> token (it will be normalized to DEFAULT_IMAGE_TOKEN).
  - Variables are optional; if provided, they'll enable LAPE injection and INPUT control-token replacement.
  - Labels are set to None (eval-like) by default to keep the test lightweight. If you want to test LAPE soft-label
    loss, adapt this script to construct labels (e.g., via your dataset/collator pipeline) and pass them to forward().
"""

import argparse
import os
import sys
from collections import defaultdict

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-root", type=str, default="VILA-ST", choices=["VILA-ST", "LLaVA-ST"], help="which package tree to import llava.* from")
    parser.add_argument("--model-path", type=str, required=True, help="Path to pretrained (merged) NVILA/LLaVA model dir")
    parser.add_argument("--model-name", type=str, default="llava_llama", help="Just for naming; not used to branch here")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file")
    parser.add_argument("--prompt", type=str, required=True, help="Conversation text, must include <image> once")
    parser.add_argument("--temporal-input", type=float, default=None, help="Optional TEMP-INPUT location in [0,1]")
    parser.add_argument("--temporal-output", type=float, default=None, help="Optional TEMP-OUTPUT target in [0,1]")
    parser.add_argument("--height-input", type=float, default=None)
    parser.add_argument("--width-input", type=float, default=None)
    parser.add_argument("--height-output", type=float, default=None)
    parser.add_argument("--width-output", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_dir, args.pkg_root))

    # Late imports from chosen package tree
    from llava.model.builder import load_pretrained_model
    from llava.utils.tokenizer import tokenize_conversation
    from llava.utils.media import extract_media
    from llava.mm_utils import process_images
    from llava.constants import DEFAULT_IMAGE_TOKEN

    print(f"Importing llava.* from: {os.path.join(repo_dir, args.pkg_root)}")
    print(f"Loading model from: {args.model_path}")

    # Load tokenizer and model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name="llava_llama",
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.to(args.device)
    model.eval()

    # Build a single-turn conversation: one user message containing one <image>
    user_text = args.prompt.replace("<image>", DEFAULT_IMAGE_TOKEN)
    conversation = [{"from": "human", "value": user_text}]

    # Prepare variables (optional)
    variables = {
        "temporal_input_locations": [],
        "temporal_output_locations": [],
        "spatial_height_input_locations": [],
        "spatial_height_output_locations": [],
        "spatial_width_input_locations": [],
        "spatial_width_output_locations": [],
    }
    if args.temporal_input is not None:
        variables["temporal_input_locations"].append(float(args.temporal_input))
    if args.temporal_output is not None:
        variables["temporal_output_locations"].append(float(args.temporal_output))
    if args.height_input is not None:
        variables["spatial_height_input_locations"].append(float(args.height_input))
    if args.width_input is not None:
        variables["spatial_width_input_locations"].append(float(args.width_input))
    if args.height_output is not None:
        variables["spatial_height_output_locations"].append(float(args.height_output))
    if args.width_output is not None:
        variables["spatial_width_output_locations"].append(float(args.width_output))

    # Extract media from the conversation (the extract_media expects list of turns)
    media = extract_media(conversation, config=model.config)
    # Overwrite the detected image path with the provided file if needed
    media["image"] = [args.image]

    # Preprocess the image using the same helper as generation
    images = process_images(media["image"], image_processor, model.config).half().to(args.device)
    media = {"image": [img for img in images]}  # list of tensors

    # Tokenize conversation
    input_ids = tokenize_conversation(conversation, tokenizer, add_generation_prompt=False).to(args.device).unsqueeze(0)

    # Build media_config: minimal required fields
    media_config = defaultdict(dict)
    media_config["image"]["block_sizes"] = [None]  # single image

    # Run a forward pass without labels (eval-like)
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            variables=[variables],
            labels=None,
        )

    logits = out.logits if hasattr(out, "logits") else out[0]
    print("Forward done.")
    print(f"logits shape: {tuple(logits.shape)}  (batch, seq, vocab)")

    # Optional: quick generation to see text output (no variables needed for generation)
    try:
        gen_ids = model.generate(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
        )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print("\nGenerated:\n", text)
    except Exception as e:
        print("Generation failed (this is optional):", str(e))


if __name__ == "__main__":
    main()
