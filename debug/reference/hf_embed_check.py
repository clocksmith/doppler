#!/usr/bin/env python3
"""
Compare embeddings and layer outputs between HuggingFace and DOPPLER.

Usage:
    python hf_embed_check.py [--model MODEL_ID] [--prompt "TEXT"]

Requires: pip install torch transformers
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Check embeddings against HuggingFace")
    parser.add_argument("--model", "-m", default="google/gemma-2-2b-it", help="HuggingFace model ID")
    parser.add_argument("--prompt", "-p", default="The color of the sky is", help="Prompt text")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs['input_ids']

    print(f"\nPrompt: {args.prompt}")
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
