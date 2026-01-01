#!/usr/bin/env python3
"""
Verify RoPE (Rotary Position Embedding) values match between HF and DOPPLER.

Usage:
    python hf_rope_check.py [--model MODEL_ID] [--pos POSITION] [--dim HEAD_DIM]

Requires: pip install torch transformers
"""

import argparse
import math
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
    parser = argparse.ArgumentParser(description="Check RoPE values")
    parser.add_argument("--model", "-m", default="google/gemma-2-2b-it", help="HuggingFace model ID")
    parser.add_argument("--pos", "-p", type=int, default=6, help="Position to check")
    parser.add_argument("--dim", "-d", type=int, default=256, help="Head dimension")
    parser.add_argument("--theta", "-t", type=float, default=10000.0, help="RoPE theta")
    args = parser.parse_args()

    print(f"RoPE Check for position {args.pos}, head_dim={args.dim}, theta={args.theta}")

    # Compute frequencies
    freqs = compute_rope_freqs(args.dim, args.theta)
    print(f"\nFrequencies (first 8): {freqs[:8].tolist()}")

    # Compute angles at position
    angles = args.pos * freqs
    print(f"Angles at pos {args.pos} (first 8): {angles[:8].tolist()}")

    # Compute cos/sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    print(f"Cos at pos {args.pos} (first 8): {cos[:8].tolist()}")
    print(f"Sin at pos {args.pos} (first 8): {sin[:8].tolist()}")

    # Test with a sample vector
    test_vec = torch.randn(args.dim)
    test_vec[0] = 1.0
    test_vec[1] = 0.0
    print(f"\nTest vector (first 8): {test_vec[:8].tolist()}")

    rotated = apply_rope_to_vector(test_vec, args.pos, freqs)
    print(f"After RoPE (first 8): {rotated[:8].tolist()}")

    # Also load model and check actual RoPE config
    print(f"\nLoading model to verify config: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
    config = model.config

    print(f"\nModel RoPE config:")
    print(f"  rope_theta: {getattr(config, 'rope_theta', 'not set')}")
    print(f"  head_dim: {config.head_dim}")
    if hasattr(config, 'rope_scaling'):
        print(f"  rope_scaling: {config.rope_scaling}")


if __name__ == "__main__":
    main()
