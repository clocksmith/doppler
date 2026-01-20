#!/usr/bin/env python3
"""
Full attention debug: traces all intermediate values through layer 0.

Outputs: input_norm -> Q/K/V projections -> RoPE -> attention scores -> output

Usage:
    python hf_attn_debug.py <config.json>

Config (JSON):
    {
      "model": "google/gemma-2-2b-it",
      "prompt": "The color of the sky is",
      "layer": 0
    }

Requires: pip install torch transformers
"""

import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python hf_attn_debug.py <config.json>")
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


def require_int(config, key):
    value = config.get(key)
    if not isinstance(value, int):
        raise SystemExit(f'Config "{key}" must be an integer')
    return value


def main():
    config = load_config()
    model_id = require_string(config, "model")
    prompt = require_string(config, "prompt")
    layer_index = require_int(config, "layer")

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    num_tokens = input_ids.shape[1]

    print(f"\nPrompt: {prompt}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Num tokens: {num_tokens}")

    # Get model config
    config = model.config
    print(f"\nModel config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.head_dim}")

    layer = model.model.layers[layer_index]
    attn = layer.self_attn

    # Get scaled embeddings as input
    raw_embeddings = model.model.embed_tokens(input_ids)
    hidden_states = raw_embeddings * (config.hidden_size ** 0.5)
    print(f"\nInput hidden states (last token first 8): {hidden_states[0, -1, :8].tolist()}")

    # Apply input norm
    normed = layer.input_layernorm(hidden_states)
    print(f"After input_norm (last token first 8): {normed[0, -1, :8].tolist()}")
    print(f"After input_norm (token 0 first 8): {normed[0, 0, :8].tolist()}")

    # Q, K, V projections
    with torch.no_grad():
        q = attn.q_proj(normed)
        k = attn.k_proj(normed)
        v = attn.v_proj(normed)

    print(f"\nQ projection (last token first 8): {q[0, -1, :8].tolist()}")
    print(f"K projection (last token first 8): {k[0, -1, :8].tolist()}")
    print(f"V projection (last token first 8): {v[0, -1, :8].tolist()}")

    print(f"\nQ projection (token 0 first 8): {q[0, 0, :8].tolist()}")
    print(f"K projection (token 0 first 8): {k[0, 0, :8].tolist()}")
    print(f"V projection (token 0 first 8): {v[0, 0, :8].tolist()}")

    # Check for errors vs expected
    print(f"\nPer-token maxAbs:")
    for t in range(num_tokens):
        q_max = q[0, t].abs().max().item()
        k_max = k[0, t].abs().max().item()
        v_max = v[0, t].abs().max().item()
        print(f"  Token {t}: Q_maxAbs={q_max:.4f}, K_maxAbs={k_max:.4f}, V_maxAbs={v_max:.4f}")


if __name__ == "__main__":
    main()
