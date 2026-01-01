#!/usr/bin/env python3
"""
Dump weight values for Q/K/V/O projections from HuggingFace model.

Usage:
    python hf_weights.py [--model MODEL_ID] [--layer LAYER] [--proj PROJ]

Requires: pip install torch transformers
"""

import argparse
import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Dump projection weights")
    parser.add_argument("--model", "-m", default="google/gemma-2-2b-it", help="HuggingFace model ID")
    parser.add_argument("--layer", "-l", type=int, default=0, help="Layer index")
    parser.add_argument("--proj", "-p", choices=["q", "k", "v", "o", "all"], default="all", help="Which projection")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")

    layer = model.model.layers[args.layer]
    attn = layer.self_attn

    projections = {
        "q": attn.q_proj,
        "k": attn.k_proj,
        "v": attn.v_proj,
        "o": attn.o_proj,
    }

    projs_to_show = list(projections.keys()) if args.proj == "all" else [args.proj]

    for name in projs_to_show:
        proj = projections[name]
        weight = proj.weight.data  # [out_features, in_features]

        print(f"\n{name.upper()}_proj weight shape: {weight.shape}")
        print(f"{name.upper()}_proj weight[0, :8] (first row, first 8 cols):")
        print(f"  {weight[0, :8].tolist()}")
        print(f"{name.upper()}_proj weight[:8, 0] (first 8 rows, first col):")
        print(f"  {weight[:8, 0].tolist()}")

        # Some specific indices for comparison
        if weight.shape[1] > 100:
            print(f"{name.upper()}_proj weight[0, 100]: {weight[0, 100].item():.6f}")


if __name__ == "__main__":
    main()
