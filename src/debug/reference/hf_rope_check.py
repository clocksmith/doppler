#!/usr/bin/env python3
"""
Verify RoPE (Rotary Position Embedding) values match between HF and DOPPLER.

Usage:
    python hf_rope_check.py <config.json>

Config (JSON):
    {
      "model": "google/gemma-2-2b-it",
      "pos": 6,
      "dim": 256,
      "theta": 10000.0
    }

Requires: pip install torch transformers
"""

import json
import math
import sys
import torch
from transformers import AutoModelForCausalLM


def compute_rope_freqs(head_dim: int, theta: float = 10000.0):
    """Compute RoPE frequencies for a given head dimension."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    return freqs


def apply_rope_to_vector(x: torch.Tensor, pos: int, freqs: torch.Tensor):
    """Apply RoPE to a single vector at a given position."""
    # x shape: [head_dim]
    angles = pos * freqs
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Split into pairs
    x_even = x[0::2]
    x_odd = x[1::2]

    # Apply rotation
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    # Interleave back
    out = torch.zeros_like(x)
    out[0::2] = out_even
    out[1::2] = out_odd

    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python hf_rope_check.py <config.json>")
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise SystemExit("Config must be a JSON object")

    model_id = config.get("model")
    pos = config.get("pos")
    dim = config.get("dim")
    theta = config.get("theta")

    if not isinstance(model_id, str) or not model_id.strip():
        raise SystemExit('Config "model" must be a non-empty string')
    if not isinstance(pos, int):
        raise SystemExit('Config "pos" must be an integer')
    if not isinstance(dim, int):
        raise SystemExit('Config "dim" must be an integer')
    if not isinstance(theta, (int, float)):
        raise SystemExit('Config "theta" must be a number')

    print(f"RoPE Check for position {pos}, head_dim={dim}, theta={theta}")

    # Compute frequencies
    freqs = compute_rope_freqs(dim, theta)
    print(f"\nFrequencies (first 8): {freqs[:8].tolist()}")

    # Compute angles at position
    angles = pos * freqs
    print(f"Angles at pos {pos} (first 8): {angles[:8].tolist()}")

    # Compute cos/sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    print(f"Cos at pos {pos} (first 8): {cos[:8].tolist()}")
    print(f"Sin at pos {pos} (first 8): {sin[:8].tolist()}")

    # Test with a sample vector
    test_vec = torch.randn(dim)
    test_vec[0] = 1.0
    test_vec[1] = 0.0
    print(f"\nTest vector (first 8): {test_vec[:8].tolist()}")

    rotated = apply_rope_to_vector(test_vec, pos, freqs)
    print(f"After RoPE (first 8): {rotated[:8].tolist()}")

    # Also load model and check actual RoPE config
    print(f"\nLoading model to verify config: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    config = model.config

    print(f"\nModel RoPE config:")
    print(f"  rope_theta: {getattr(config, 'rope_theta', 'not set')}")
    print(f"  head_dim: {config.head_dim}")
    if hasattr(config, 'rope_scaling'):
        print(f"  rope_scaling: {config.rope_scaling}")


if __name__ == "__main__":
    main()
