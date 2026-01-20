# Reference Model Debug Scripts

These scripts run the original SafeTensor model weights via PyTorch/transformers to compare against DOPPLER's quantized inference. Use these to isolate whether bugs are in DOPPLER's kernels, weight loading, or architecture implementation. They are config-only and accept a single JSON config path (no flags).

## Setup

```bash
pip install torch transformers
```

## Scripts

### hf_embed_check.py
Compare embeddings and layer 0 outputs.

```json
// tmp-hf-embed.json
{
  "model": "google/gemma-2-2b-it",
  "prompt": "The color of the sky is"
}
```

```bash
python hf_embed_check.py ./tmp-hf-embed.json
```

### hf_attn_debug.py
Full attention debug: traces input_norm -> Q/K/V projections.

```json
// tmp-hf-attn.json
{
  "model": "google/gemma-2-2b-it",
  "prompt": "The color of the sky is",
  "layer": 0
}
```

```bash
python hf_attn_debug.py ./tmp-hf-attn.json
```

### hf_weights.py
Dump Q/K/V/O projection weights for comparison.

```json
// tmp-hf-weights.json
{
  "model": "google/gemma-2-2b-it",
  "layer": 0,
  "proj": "v"
}
```

```bash
python hf_weights.py ./tmp-hf-weights.json
```

### hf_rope_check.py
Verify RoPE frequencies and rotations.

```json
// tmp-hf-rope.json
{
  "model": "google/gemma-2-2b-it",
  "pos": 6,
  "dim": 256,
  "theta": 10000.0
}
```

```bash
python hf_rope_check.py ./tmp-hf-rope.json
```

### hf_layer_out.py
Compare hidden states at specific layers.

```json
// tmp-hf-layer.json
{
  "model": "google/gemma-2-2b-it",
  "prompt": "The color of the sky is",
  "layers": [0, 12, 25],
  "token": -1
}
```

```bash
python hf_layer_out.py ./tmp-hf-layer.json
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
