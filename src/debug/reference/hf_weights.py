#!/usr/bin/env python3
"""
Dump weight values for Q/K/V/O projections from HuggingFace model.

Usage:
    python hf_weights.py <config.json>

Config (JSON):
    {
      "model": "google/gemma-2-2b-it",
      "layer": 0,
      "proj": "all"
    }

Requires: pip install torch transformers
"""

import json
import sys
import torch
from transformers import AutoModelForCausalLM


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python hf_weights.py <config.json>")
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise SystemExit("Config must be a JSON object")

    model_id = config.get("model")
    layer_index = config.get("layer")
    proj = config.get("proj")

    if not isinstance(model_id, str) or not model_id.strip():
        raise SystemExit('Config "model" must be a non-empty string')
    if not isinstance(layer_index, int):
        raise SystemExit('Config "layer" must be an integer')
    if proj not in ["q", "k", "v", "o", "all"]:
        raise SystemExit('Config "proj" must be one of: q, k, v, o, all')

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")

    layer = model.model.layers[layer_index]
    attn = layer.self_attn

    projections = {
        "q": attn.q_proj,
        "k": attn.k_proj,
        "v": attn.v_proj,
        "o": attn.o_proj,
    }

    projs_to_show = list(projections.keys()) if proj == "all" else [proj]

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
