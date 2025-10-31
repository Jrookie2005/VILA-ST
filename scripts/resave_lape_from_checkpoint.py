#!/usr/bin/env python3
"""
Resave LAPE embeddings (lape_embeddings.bin) from a Trainer checkpoint.

Use this if your previously saved lape_embeddings.bin was empty or invalid due to ZeRO partitioning.

Examples:
  python VILA-ST/scripts/resave_lape_from_checkpoint.py \
      --src runs/train/nvila-8b-AgroSFT-lape/model \
      --dst runs/train/nvila-8b-AgroSFT-lape/model

It will auto-detect the latest checkpoint-* under --src and extract LAPE weights
from pytorch_model.bin or model.safetensors into lape_embeddings.bin.

Optionally, you can point --ckpt to a specific checkpoint directory.
"""
import argparse
import glob
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
import json
import shutil
import subprocess

import torch

try:
    from safetensors.torch import load_file as safe_load
except Exception:
    safe_load = None


LAPE_MODULES = [
    "spatial_height_input_embeddings",
    "spatial_height_output_embeddings",
    "spatial_width_input_embeddings",
    "spatial_width_output_embeddings",
    "temporal_input_embeddings",
    "temporal_output_embeddings",
]


def find_latest_checkpoint(model_dir: str) -> str | None:
    pattern = os.path.join(model_dir, "checkpoint-*")
    cks = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not cks:
        return None
    def step_num(p: str) -> int:
        m = re.search(r"checkpoint-(\d+)$", p)
        return int(m.group(1)) if m else -1
    cks.sort(key=step_num, reverse=True)
    return cks[0]


def _load_single_state_dict_file(path: str) -> dict:
    ext = os.path.basename(path)
    if ext.endswith(".safetensors") and safe_load is not None:
        return safe_load(path)
    return torch.load(path, map_location="cpu")


def load_state_dict(ckpt_dir: str) -> dict:
    """Load a (possibly sharded) state dict from a checkpoint directory.

    Supported patterns:
      - model.safetensors
      - pytorch_model.bin
      - pytorch_model.bin.index.json + shards
      - model-*.safetensors shards (with or without index)
      - pytorch_model-*.bin shards (with or without index)
    """
    # 1) Direct single-file
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(st_path) and safe_load is not None:
        return safe_load(st_path)
    pt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(pt_path):
        return torch.load(pt_path, map_location="cpu")

    # 2) Index-based shards
    index_path = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        # Build file -> keys mapping
        files = set(weight_map.values())
        merged = {}
        for fname in files:
            fpath = os.path.join(ckpt_dir, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Shard file missing: {fpath}")
            shard_sd = _load_single_state_dict_file(fpath)
            merged.update(shard_sd)
        return merged

    # 3) Shards without index (best-effort merge)
    shard_bins = sorted(glob.glob(os.path.join(ckpt_dir, "pytorch_model-*.bin")))
    shard_safes = sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
    merged = {}
    loaded_any = False
    for fpath in shard_bins + shard_safes:
        try:
            shard_sd = _load_single_state_dict_file(fpath)
            merged.update(shard_sd)
            loaded_any = True
        except Exception:
            pass
    if loaded_any:
        return merged

    # 4) Try to materialize full weights using deepspeed zero_to_fp32.py if present
    z2fp32 = os.path.join(ckpt_dir, "zero_to_fp32.py")
    if os.path.isfile(z2fp32):
        out_bin = os.path.join(ckpt_dir, "pytorch_model.bin")
        try:
            print(f"[INFO] Found zero_to_fp32.py. Materializing full weights to {out_bin} ...")
            subprocess.run([sys.executable, z2fp32, ckpt_dir, out_bin], check=True)
            if os.path.isfile(out_bin):
                return torch.load(out_bin, map_location="cpu")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] zero_to_fp32.py failed: {e}")

    raise FileNotFoundError(
        f"No state dict files found in {ckpt_dir} (model.safetensors / pytorch_model.bin / shards / zero_to_fp32.py)"
    )


def extract_lape_substate(sd: dict) -> dict:
    """Extract sub-state dicts for each LAPE module, tolerant to common prefixes."""
    lape_sd = OrderedDict()
    prefixes = [
        "",               # direct
        "model.",         # sometimes hf adds model.
        "module.",        # DDP
        "model.model.",   # nested
    ]
    for mod in LAPE_MODULES:
        # Build candidate keys for this module
        sub = OrderedDict()
        # we expect keys like f"{mod}.weight" / f"{mod}.bias" or Linear weights
        for k, v in sd.items():
            # Try strip any known prefix then match module name
            base = k
            for p in prefixes:
                if base.startswith(p):
                    base = base[len(p):]
            if base.startswith(mod + "."):
                sub_key = base[len(mod) + 1 :]
                sub[sub_key] = v
        if sub:
            lape_sd[mod] = sub
        else:
            print(f"[WARN] Could not find weights for {mod} in checkpoint state dict. Skipping.")
    return lape_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to the saved model folder (contains config.json, llm, mm_projector, vision_tower)")
    ap.add_argument("--dst", required=False, default=None, help="Output model folder to write lape_embeddings.bin (default: same as --src)")
    ap.add_argument("--ckpt", required=False, default=None, help="Explicit checkpoint-* folder; if omitted, auto-detect latest under --src")
    ap.add_argument("--force", action="store_true", help="Force rebuild from full state dict (ignore existing lape_embeddings.bin in checkpoint)")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.dst or model_dir)

    ckpt_dir = args.ckpt
    if ckpt_dir is None:
        ckpt_dir = find_latest_checkpoint(model_dir)
        if ckpt_dir is None:
            # If no checkpoint in model_dir, try searching parent run folder recursively for latest checkpoint
            parent = os.path.abspath(os.path.join(model_dir, os.pardir, os.pardir))
            candidates = sorted(
                [p for p in glob.glob(os.path.join(parent, "**", "checkpoint-*"), recursive=True) if os.path.isdir(p)],
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )
            ckpt_dir = candidates[0] if candidates else None
            if ckpt_dir is None:
                print(f"[ERROR] No checkpoint-* folder found under {model_dir} or parent run folder {parent}")
                sys.exit(1)
    ckpt_dir = os.path.abspath(ckpt_dir)

    print(f"Source model dir: {model_dir}")
    print(f"Checkpoint dir  : {ckpt_dir}")
    print(f"Output dir      : {out_dir}")

    # Helper to judge if a lape_embeddings.bin has non-empty tensors
    def _lape_file_is_nonempty(path: str) -> bool:
        try:
            sd = torch.load(path, map_location="cpu")
            if not isinstance(sd, dict):
                return False
            nonempty = False
            for mod, sub in sd.items():
                if isinstance(sub, dict):
                    w = sub.get("weight", None)
                    if w is not None and hasattr(w, "numel") and w.numel() > 0:
                        nonempty = True
                        break
            return nonempty
        except Exception:
            return False

    # Fast path: if checkpoint already contains lape_embeddings.bin and it's non-empty, just copy it
    ckpt_lape = os.path.join(ckpt_dir, "lape_embeddings.bin")
    out_path = os.path.join(out_dir, "lape_embeddings.bin")
    if (not args.force) and os.path.isfile(ckpt_lape) and _lape_file_is_nonempty(ckpt_lape):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(ckpt_lape, out_path)
        print(f"[INFO] Found LAPE file in checkpoint. Copied to: {out_path}")
        return
    elif (not args.force) and os.path.isfile(ckpt_lape):
        print("[INFO] Found LAPE file in checkpoint but it appears empty (ZeRO shard). Will rebuild from full state dict.")
    elif args.force:
        print("[INFO] --force set. Will rebuild from full state dict.")

    # Try load state dict from checkpoint dir
    try:
        sd = load_state_dict(ckpt_dir)
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("[INFO] Will scan for shard files one level above checkpoint dir as a fallback.")
        above = os.path.abspath(os.path.join(ckpt_dir, os.pardir))
        sd = None
        for probe in [ckpt_dir, above]:
            try:
                sd = load_state_dict(probe)
                ckpt_dir = probe
                break
            except Exception:
                continue
        if sd is None:
            print("[ERROR] Unable to locate any state dict files (single or sharded).")
            sys.exit(1)
    lape_sd = extract_lape_substate(sd)

    if not lape_sd:
        print("[ERROR] No LAPE weights extracted. Nothing to save.")
        sys.exit(2)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(lape_sd, out_path)
    print(f"Saved LAPE embeddings to: {out_path}")


if __name__ == "__main__":
    main()
