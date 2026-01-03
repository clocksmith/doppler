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
# Standard conversion with Q4_K_M quantization (recommended for most models)
npx tsx src/converter/node-converter.ts INPUT_PATH models/OUTPUT_NAME --quantize q4_k_m

# From GGUF file (already quantized)
npx tsx src/converter/node-converter.ts INPUT.gguf models/OUTPUT_NAME

# Keep full FP16 precision (for debugging quantization issues)
npx tsx src/converter/node-converter.ts INPUT_PATH models/OUTPUT_NAME --quantize f16

# Multimodal to text-only (Gemma 3, PaliGemma, LLaVA)
npx tsx src/converter/node-converter.ts INPUT_PATH models/OUTPUT_NAME --text-only --quantize q4_k_m

# Create tiny test fixture for development
npx tsx src/converter/node-converter.ts --test ./test-model
```

## Conversion Options

| Flag | Description | When to Use |
|------|-------------|-------------|
| `--quantize q4_k_m` | 4-bit quantization | Default, good quality/size tradeoff |
| `--quantize f16` | Full FP16 precision | Debugging, reference testing |
| `--quantize-embeddings` | Also quantize embeddings | When size is critical |
| `--text-only` | Strip vision/audio towers | Multimodal → text-only |
| `--shard-size <mb>` | Shard size (default: 64) | Tune for network/storage |
| `--model-id <id>` | Override model ID | Custom naming (default uses quant tag) |
| `--verbose` | Detailed progress | Debugging conversion |
| `--fast` | Pre-load shards | Faster but uses more RAM |

## Post-Conversion Verification (Mandatory)

Never report conversion complete until verified:

```bash
# 1. Check manifest exists and looks correct
cat models/OUTPUT_NAME/manifest.json | grep -E "\"architecture\"|\"tensorCount\"|\"num_hidden_layers\"|\"quantizationInfo\""

# 2. Check shard files exist
ls -lh models/OUTPUT_NAME/

# 3. Test inference actually works
npm run debug -- -m OUTPUT_NAME 2>&1 | grep -E "Done|Output|Error"
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
| Large output size | No quantization flag | Add `--quantize q4_k_m` |
| Missing tensors | Multimodal model | Add `--text-only` flag |
| Inference fails after convert | Weight layout issue | Try `--quantize f16` to isolate |

## Naming Convention (Derived from Conversion Flags)

By default the converter appends a quantization tag to `modelId`:

- Weights + embeddings tag: `w<weights>-emb<embeddings>`
- Example: `gemma-2-2b-it-wq4_k_m-embf16`

Override with `--model-id` if you need a custom name.

## Reference Files

For detailed information, consult these files:

- **Model support matrix**: `docs/plans/TARGET_MODELS.md`
- **RDRR format spec**: `docs/design/RDRR_FORMAT.md`
- **Converter source**: `src/converter/node-converter.ts`
- **Troubleshooting**: `docs/DOPPLER-TROUBLESHOOTING.md`

## Related Skills

- Use `doppler-debug` if converted model produces wrong output
- Use `doppler-benchmark` to measure converted model performance
