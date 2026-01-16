---
name: doppler-convert
description: Convert GGUF or SafeTensors models to DOPPLER's RDRR format and verify they load in the browser runtime. Use when adding a new model, re-quantizing weights, or creating test fixtures. (project)
---

# DOPPLER Convert Skill

Use this skill to convert models from GGUF/SafeTensors to DOPPLER's RDRR format.

## Pre-Flight Checks (Mandatory)

Before starting conversion, verify:

```bash
# 1. Verify source model exists
ls -lh INPUT_PATH

# 2. Check available disk space (need 2x model size)
df -h .

# 3. For HuggingFace models, check the cache location
ls ~/.cache/huggingface/hub/models--ORG--MODEL/snapshots/
```

Ensure disk space; insufficient space can leave corrupted output.

## Conversion Commands

```bash
# Standard conversion with Q4K weights (recommended)
npx tsx src/converter/node-converter.js INPUT_PATH models/OUTPUT_NAME -w q4k

# From GGUF file (already quantized)
npx tsx src/converter/node-converter.js INPUT.gguf models/OUTPUT_NAME

# Keep full FP16 precision (for debugging quantization issues)
npx tsx src/converter/node-converter.js INPUT_PATH models/OUTPUT_NAME -w f16

# Explicit embeddings quantization
npx tsx src/converter/node-converter.js INPUT_PATH models/OUTPUT_NAME -w q4k -e f16

# Multimodal to text-only (Gemma 3, PaliGemma, LLaVA)
npx tsx src/converter/node-converter.js INPUT_PATH models/OUTPUT_NAME --text-only -w q4k

# Use a converter config file (JSON)
npx tsx src/converter/node-converter.js INPUT_PATH models/OUTPUT_NAME --config ./converter-config.json

# Create tiny test fixture for development
npx tsx src/converter/node-converter.js --test ./test-model
```

## Conversion Options

### Quantization Options

| Flag | Description | When to Use |
|------|-------------|-------------|
| `--weight-quant, -w <type>` | Weight quantization: q4k, f16, f32 | Main model weights |
| `--embed-quant, -e <type>` | Embedding quantization | Keep embeddings at higher precision |
| `--head-quant <type>` | LM head quantization | Keep output head at higher precision |
| `--vision-quant <type>` | Vision encoder quantization | Multimodal models |
| `--audio-quant <type>` | Audio encoder quantization | Speech models |
| `--projector-quant <type>` | Cross-modal projector | Multimodal models |

### Runtime Hints (stored in manifest, not filename)

| Flag | Description | When to Use |
|------|-------------|-------------|
| `--compute-precision <p>` | f16, f32, or auto (default: f16) | Suggest compute precision |

### General Options

| Flag | Description | When to Use |
|------|-------------|-------------|
| `--config, -c <path>` | Load JSON converter config | Reuse consistent settings |
| `--text-only` | Strip vision/audio towers | Multimodal → text-only |
| `--shard-size <mb>` | Shard size (default: 64) | Tune for network/storage |
| `--model-id <id>` | Override model ID | Custom naming |
| `--verbose` | Detailed progress | Debugging conversion |
| `--fast` | Pre-load shards | Faster but uses more RAM |

## Post-Conversion Verification (Mandatory)

Never report conversion complete until verified:

```bash
# 1. Check manifest exists and looks correct
cat models/OUTPUT_NAME/manifest.json | grep -E "\"architecture\"|\"tensorCount\"|\"num_hidden_layers\"|\"quantizationInfo\""

# 2. Check shard files exist
ls -lh models/OUTPUT_NAME/

# 3. Test inference actually works (uses DOPPLER completion signals)
npm run debug -- --config debug -m OUTPUT_NAME 2>&1 | grep -E "DOPPLER:DONE|DOPPLER:ERROR"
```

If verification fails:
1. Delete the corrupted output: `rm -rf models/OUTPUT_NAME`
2. Check error message for cause
3. Re-run with `--verbose` to diagnose

## Supported Input Formats

| Format | Source | Example Path |
|--------|--------|--------------|
| HuggingFace directory | `transformers` download | `~/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/abc123/` |
| SafeTensors files | Direct download | `./model.safetensors` or directory with multiple |
| GGUF file | llama.cpp conversion | `./model-Q4_K_M.gguf` |
| Sharded index | Large HF models | Directory with `model.safetensors.index.json` |

## Common Issues and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "Unknown architecture" | Unsupported model type | Check `docs/plans/TARGET_MODELS.md` |
| "ENOSPC" or disk full | Insufficient space | Free up 2x model size |
| Missing config values | Incomplete HF config | Converter infers from shapes |
| Large output size | No quantization flag | Add `-w q4k` |
| Missing tensors | Multimodal model | Add `--text-only` flag |
| Inference fails after convert | Weight layout issue | Try `-w f16` to isolate |

## Naming Convention

DOPPLER uses a concise naming convention that describes **storage only** (not runtime):

```
{model-name}-w{weights}[-e{embeddings}][-h{head}][-v{vision}][-a{audio}][-p{projector}]
```

### Component Prefixes

| Prefix | Component | When Included |
|--------|-----------|---------------|
| `w` | Weights | Always (required) |
| `e` | Embeddings | When different from weights |
| `h` | Head | When different from embeddings |
| `v` | Vision | Multimodal with vision encoder |
| `a` | Audio | Speech models |
| `p` | Projector | Multimodal models |

### Quantization Tokens

| Token | Description | Token | Description |
|-------|-------------|-------|-------------|
| `q4k` | Q4_K_M block quant | `f16` | Float16 |
| `q6k` | Q6_K block quant | `bf16` | BFloat16 |
| `q8_0` | Q8_0 quant | `f32` | Float32 |

### Examples

| Command | Resulting Model ID |
|---------|-------------------|
| `-w q4k` | `gemma-2b-wq4k` (if embeds same) |
| `-w q4k` (bf16 source) | `gemma-2b-wq4k-ebf16` |
| `-w q4k -e f16` | `gemma-2b-wq4k-ef16` |
| `-w q4k -e f16 --head-quant f32` | `gemma-2b-wq4k-ef16-hf32` |
| `-w q4k --vision-quant f16` | `qwen2-vl-7b-wq4k-vf16` |

Override with `--model-id` if you need a custom name.

## Reference Files

For detailed information, consult these files:

- **Model support matrix**: `docs/plans/TARGET_MODELS.md`
- **RDRR format spec**: `docs/spec/RDRR_FORMAT.md`
- **Converter source**: `src/converter/node-converter/index.js`
- **Troubleshooting**: `docs/DOPPLER-TROUBLESHOOTING.md`

## Related Skills

- Use `doppler-debug` if converted model produces wrong output
- Use `doppler-benchmark` to measure converted model performance
