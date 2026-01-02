---
name: doppler-convert
description: Convert GGUF or SafeTensors models to RDRR format and test them in DOPPLER. Use when the user wants to add a new model, convert weights, or verify model loading. (project)
---

# Pre-Flight Checks (Mandatory)

```bash
# Verify source exists
ls -lh INPUT_PATH

# Check disk space - need 2x model size free
df -h .
```

# Conversion Commands

```bash
# From HuggingFace directory with quantization (recommended)
npx tsx tools/convert-cli.ts INPUT_DIR models/OUTPUT_NAME --quantize q4_k_m

# From GGUF file
npx tsx tools/convert-cli.ts INPUT.gguf models/OUTPUT_NAME

# Multimodal to text-only (strips vision tower)
npx tsx tools/convert-cli.ts INPUT_DIR models/OUTPUT_NAME --text-only --quantize q4_k_m

# Keep full precision for debugging
npx tsx tools/convert-cli.ts INPUT_DIR models/OUTPUT_NAME --quantize f16
```

# Options Reference

| Flag | Description |
|------|-------------|
| `--quantize q4_k_m` | Quantize to Q4_K_M (recommended) |
| `--quantize f16` | Keep FP16 precision |
| `--text-only` | Extract text model from multimodal |
| `--shard-size <mb>` | Shard size in MB (default: 64) |
| `--verbose` | Show detailed progress |

# Post-Conversion Verification (Mandatory)

Never mark conversion complete until verified:

```bash
# Check manifest exists and has correct layer count
cat models/OUTPUT_NAME/manifest.json | grep -E "num_layers|tensor_count"

# Test inference actually works
npm run debug -- -m OUTPUT_NAME 2>&1 | grep -E "Done|Output|Error"
```

If verification fails, delete corrupted output and report error.

# Common Issues

| Issue | Fix |
|-------|-----|
| "Unknown architecture" | Check MODEL_SUPPORT.md |
| Large output size | Add `--quantize q4_k_m` |
| Missing tensors | Add `--text-only` for multimodal |

# Related Skills

Use `doppler-debug` if inference fails. Use `doppler-benchmark` to measure performance.
