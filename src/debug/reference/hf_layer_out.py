#!/usr/bin/env python3
"""
Compare layer outputs at each layer between HuggingFace and DOPPLER.

Usage:
    python hf_layer_out.py [--model MODEL_ID] [--prompt "TEXT"] [--layers 0,5,10]

Requires: pip install torch transformers
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Compare layer outputs")
    parser.add_argument("--model", "-m", default="google/gemma-2-2b-it", help="HuggingFace model ID")
    parser.add_argument("--prompt", "-p", default="The color of the sky is", help="Prompt text")
    parser.add_argument("--layers", "-l", default="0,12,25", help="Comma-separated layer indices")
    parser.add_argument("--token", "-t", type=int, default=-1, help="Token index (-1 for last)")
    args = parser.parse_args()

    layer_indices = [int(x) for x in args.layers.split(",")]

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    num_tokens = input_ids.shape[1]
    token_idx = args.token if args.token >= 0 else num_tokens + args.token

    print(f"\nPrompt: {args.prompt}")
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
