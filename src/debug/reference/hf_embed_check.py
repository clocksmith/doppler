#!/usr/bin/env python3
"""
Compare embeddings and layer outputs between HuggingFace and DOPPLER.

Usage:
    python hf_embed_check.py <config.json>

Config (JSON):
    {
      "model": "google/gemma-2-2b-it",
      "prompt": "The color of the sky is"
    }

Requires: pip install torch transformers
"""

import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python hf_embed_check.py <config.json>")
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise SystemExit("Config must be a JSON object")
    return config


def require_string(config, key):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'Config "{key}" must be a non-empty string')
    return value


def main():
    config = load_config()
    model_id = require_string(config, "model")
    prompt = require_string(config, "prompt")

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']

    print(f"\nPrompt: {prompt}")
    print(f"Token IDs: {input_ids[0].tolist()}")

    # Get embeddings (raw, not scaled)
    raw_embeddings = model.model.embed_tokens(input_ids)
    print(f"\nRaw embeddings shape: {raw_embeddings.shape}")
    print(f"Raw embeddings (last token first 5): {raw_embeddings[0, -1, :5].tolist()}")

    # Gemma scales embeddings by sqrt(hidden_size)
    hidden_size = model.config.hidden_size
    scaled_embeddings = raw_embeddings * (hidden_size ** 0.5)
    print(f"Scaled embeddings (last token first 5): {scaled_embeddings[0, -1, :5].tolist()}")

    # Get layer 0 output via hooks
    layer0_outputs = {}

    def hook_layer0_output(module, input, output):
        layer0_outputs['output'] = output[0].detach()

    model.model.layers[0].register_forward_hook(hook_layer0_output)

    # Run forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    print(f"\nLayer 0 output (last token first 5): {layer0_outputs['output'][0, -1, :5].tolist()}")

    # Per-token comparison
    print(f"\nPer-token maxAbs comparison:")
    for t in range(inputs['input_ids'].shape[1]):
        emb_max = scaled_embeddings[0, t].abs().max().item()
        l0_max = layer0_outputs['output'][0, t].abs().max().item()
        print(f"  Token {t}: emb_maxAbs={emb_max:.2f}, L0_maxAbs={l0_max:.2f}")


if __name__ == "__main__":
    main()
