---
name: doppler-convert
description: Convert GGUF or SafeTensors models to RDRR format and test them in DOPPLER. Use when the user wants to add a new model, convert weights, or verify model loading. (project)
---

# Model Conversion Skill

Convert models to DOPPLER's RDRR format and verify they work.

## Conversion CLI

```bash
# From HuggingFace directory with quantization
npx tsx src/converter/node-converter.ts \
  ~/models/Llama-3.2-1B \
  models/llama-1b \
  --quantize q4_k_m

# From GGUF file
npx tsx src/converter/node-converter.ts \
  ~/models/model.gguf \
  models/model-name

# Multimodal to text-only (strips vision tower)
npx tsx src/converter/node-converter.ts \
  ~/models/gemma-3-4b-it \
  models/gemma-4b-text \
  --text-only --quantize q4_k_m

# Create tiny test fixture
npx tsx src/converter/node-converter.ts --test ./test-model
```

## Options

| Flag | Description |
|------|-------------|
| `--quantize q4_k_m` | Quantize to Q4_K_M (recommended) |
| `--quantize f16` | Keep as FP16 |
| `--quantize-embeddings` | Also quantize embedding table |
| `--shard-size <mb>` | Shard size in MB (default: 64) |
| `--model-id <id>` | Override model ID in manifest |
| `--text-only` | Extract only text model from multimodal |
| `--fast` | Pre-load shards (faster, more RAM) |
| `--verbose` | Show detailed progress |

## Testing the Converted Model

```bash
# Quick smoke test (does it load and generate?)
npm test -- --inference

# Debug mode (for investigating issues)
npm run debug -- --config debug

# Benchmark performance
npm run bench -- --config bench

# Or test manually in browser (requires npm start first)
# http://localhost:8080/doppler/tests/test-inference.html?model=<model-name>
```

**Config Presets:** `--config` loads runtime presets (not model presets). Use `--config debug` for verbose logging, `--config bench` for clean output.

**Log Levels:** Add `--verbose` for per-shard/layer timing, `--trace` for tensor details.

**List Presets:** `npx tsx cli/index.ts --list-presets`

## Workflow

1. **Locate source model**
   - HuggingFace cache: `~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<hash>/`
   - Local GGUF: Any `.gguf` file
   - Local SafeTensors: Directory with `*.safetensors` files

2. **Convert with appropriate options**
   - Use `--quantize q4_k_m` for smaller size
   - Use `--text-only` for multimodal models (Gemma 3, PaliGemma)
   - Use `--fast` if you have enough RAM

3. **Verify conversion output**
   - Check `manifest.json` in output directory
   - Verify tensor count and shard count
   - Check model config is correctly inferred

4. **Test in browser**
   - Start server, run E2E test
   - Check for inference errors in console
   - If issues, use `doppler-debug` skill

## Supported Input Formats

| Format | Extension | Source |
|--------|-----------|--------|
| GGUF | `.gguf` | llama.cpp |
| SafeTensors | `.safetensors` | HuggingFace |
| HF Directory | folder | HuggingFace Hub |
| Index JSON | `model.safetensors.index.json` | Sharded HF models |

## Output Structure

```
models/<model-name>/
  manifest.json       # Model metadata, tensor index
  shard_00000.bin     # Weight shards
  shard_00001.bin
  ...
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| "Unknown architecture" | Model type not recognized | Check MODEL_SUPPORT.md for supported archs |
| Config values missing | HF config incomplete | Converter infers from tensor shapes |
| Large output size | No quantization | Add `--quantize q4_k_m` |
| Missing tensors | Multimodal model | Add `--text-only` for text-only extraction |

## Reference

- Model support matrix: `docs/plans/MODEL_SUPPORT.md`
- RDRR format spec: `docs/spec/RDRR_FORMAT.md`
- Troubleshooting: `docs/DOPPLER-TROUBLESHOOTING.md`

## Related Skills

- **doppler-debug**: Debug inference issues with converted models
- **doppler-benchmark**: Measure performance of converted models

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.
