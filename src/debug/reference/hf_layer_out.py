#!/usr/bin/env python3
"""
Compare layer outputs at each layer between HuggingFace and DOPPLER.

Usage:
    python hf_layer_out.py <config.json>

Config (JSON):
    {
      "model": "google/gemma-2-2b-it",
      "prompt": "The color of the sky is",
      "layers": [0, 12, 25],
      "token": -1
    }

Requires: pip install torch transformers
"""

import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python hf_layer_out.py <config.json>")
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise SystemExit("Config must be a JSON object")

    model_id = config.get("model")
    prompt = config.get("prompt")
    layers = config.get("layers")
    token = config.get("token")

    if not isinstance(model_id, str) or not model_id.strip():
        raise SystemExit('Config "model" must be a non-empty string')
    if not isinstance(prompt, str) or not prompt.strip():
        raise SystemExit('Config "prompt" must be a non-empty string')
    if not isinstance(layers, list) or not layers:
        raise SystemExit('Config "layers" must be a non-empty array')
    if not all(isinstance(x, int) for x in layers):
        raise SystemExit('Config "layers" entries must be integers')
    if not isinstance(token, int):
        raise SystemExit('Config "token" must be an integer')

    layer_indices = layers

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    num_tokens = input_ids.shape[1]
    token_idx = token if token >= 0 else num_tokens + token

    print(f"\nPrompt: {prompt}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Checking token index: {token_idx}")

    # Run forward pass with hidden states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden]
    print(f"\nNum hidden states: {len(hidden_states)} (embed + {len(hidden_states)-1} layers)")

    for layer_idx in layer_indices:
        if layer_idx >= len(hidden_states) - 1:
            print(f"\nLayer {layer_idx}: OUT OF RANGE")
            continue

        # hidden_states[0] = embeddings, hidden_states[1] = layer 0 output, etc.
        hs = hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings

        vals = hs[0, token_idx, :8].tolist()
        max_abs = hs[0, token_idx].abs().max().item()
        mean_abs = hs[0, token_idx].abs().mean().item()

        print(f"\nLayer {layer_idx} output (token {token_idx}):")
        print(f"  First 8: {[f'{v:.4f}' for v in vals]}")
        print(f"  maxAbs: {max_abs:.4f}, meanAbs: {mean_abs:.4f}")

    # Also show BOS (token 0) for comparison
    print(f"\n--- BOS Token (position 0) for reference ---")
    for layer_idx in layer_indices:
        if layer_idx >= len(hidden_states) - 1:
            continue
        hs = hidden_states[layer_idx + 1]
        vals = hs[0, 0, :8].tolist()
        max_abs = hs[0, 0].abs().max().item()
        print(f"Layer {layer_idx} (BOS): first8={[f'{v:.4f}' for v in vals]}, maxAbs={max_abs:.4f}")


if __name__ == "__main__":
    main()
