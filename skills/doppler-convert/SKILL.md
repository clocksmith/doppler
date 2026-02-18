---
name: doppler-convert
description: Convert GGUF or SafeTensors assets into Doppler RDRR manifests/shards using the current Node command surface, then verify load + inference. (project)
---

# DOPPLER Convert Skill

Use this skill to add or re-convert models for Doppler runtime.

## Primary Conversion Commands

```bash
# Convert from Safetensors directory (or GGUF file path) via unified CLI
npm run convert -- INPUT_PATH models/local/OUTPUT_ID --model-id OUTPUT_ID

# Same conversion through direct Node helper with converter-config JSON
node tools/convert-safetensors-node.js INPUT_PATH models/local/OUTPUT_ID --model-id OUTPUT_ID --converter-config ./converter-config.json
```

Notes:
- `INPUT_PATH` can be a Safetensors directory, diffusion directory, or `.gguf` file.
- Unified CLI convert path is `tools/doppler-cli.js` -> `runNodeCommand()` -> `src/tooling/node-convert.js`.
- Browser surface is intentionally rejected for `convert`.

## Converter Config JSON (Optional)

Example:

```json
{
  "quantization": {
    "weights": "q4k",
    "embeddings": "f16",
    "lmHead": "f16",
    "q4kLayout": "row",
    "computePrecision": "f16"
  },
  "output": {
    "textOnly": false
  }
}
```

## Post-Conversion Verification (Mandatory)

```bash
# 1) Manifest exists
test -f models/local/OUTPUT_ID/manifest.json

# 2) Verify key manifest fields
jq '.modelId, .modelType, .quantization, .quantizationInfo, .inference.defaultKernelPath' models/local/OUTPUT_ID/manifest.json

# 3) Verify shards exist
ls models/local/OUTPUT_ID/shard_*.bin | wc -l

# 4) Sanity-run inference
npm run debug -- --model-id OUTPUT_ID --runtime-preset modes/debug --surface auto --json
```

## Conversion Triage Contract

When conversion quality is in question, follow `AGENTS.md` triage protocol:
1. Verify source dtypes.
2. Verify manifest `quantization` + `quantizationInfo` + default kernel path.
3. Verify shard integrity vs manifest hashes.
4. Verify sampled tensor numeric sanity source vs converted bytes.
5. Verify layer pattern semantics (`every_n` behavior).

## Canonical Files

- `tools/doppler-cli.js`
- `tools/convert-safetensors-node.js`
- `src/tooling/node-command-runner.js`
- `src/tooling/node-convert.js`
- `src/converter/core.js`
- `src/converter/conversion-plan.js`
- `docs/FORMATS.md`
- `AGENTS.md`

## Related Skills

- `doppler-debug` for runtime correctness after conversion
- `doppler-bench` for perf regressions between variants
