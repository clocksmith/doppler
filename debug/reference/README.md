# Reference Model Debug Scripts

These scripts run the original SafeTensor model weights via PyTorch/transformers to compare against DOPPLER's quantized inference. Use these to isolate whether bugs are in DOPPLER's kernels, weight loading, or architecture implementation.

## Setup

```bash
pip install torch transformers
```

## Scripts

### hf_embed_check.py
Compare embeddings and layer 0 outputs.

```bash
python hf_embed_check.py --model google/gemma-2-2b-it --prompt "The color of the sky is"
```

### hf_attn_debug.py
Full attention debug: traces input_norm -> Q/K/V projections.

```bash
python hf_attn_debug.py --model google/gemma-2-2b-it --layer 0
```

### hf_weights.py
Dump Q/K/V/O projection weights for comparison.

```bash
python hf_weights.py --model google/gemma-2-2b-it --layer 0 --proj v
```

### hf_rope_check.py
Verify RoPE frequencies and rotations.

```bash
python hf_rope_check.py --model google/gemma-2-2b-it --pos 6 --dim 256
```

### hf_layer_out.py
Compare hidden states at specific layers.

```bash
python hf_layer_out.py --model google/gemma-2-2b-it --layers 0,12,25
```

## Debugging Process

1. **Identify divergence layer**: Run `hf_layer_out.py` to find where DOPPLER diverges from reference
2. **Check embeddings**: Run `hf_embed_check.py` to verify embedding scaling
3. **Debug attention**: Run `hf_attn_debug.py` at the diverging layer
4. **Check weights**: Run `hf_weights.py` to compare dequantized weight values
5. **Verify RoPE**: Run `hf_rope_check.py` if Q/K values diverge after position 0

## Common Issues

- **Embedding scale**: Gemma uses `sqrt(hidden_size)` scaling
- **RMSNorm offset**: Gemma uses `(1 + weight) * x` formula
- **Q4K dequant**: Check for repeated values (indexing bug)
- **Attention softcapping**: Gemma 2 uses `attn_logit_soft_capping=50`
