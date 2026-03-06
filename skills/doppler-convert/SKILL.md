---
name: doppler-convert
description: Convert GGUF or SafeTensors assets into Doppler RDRR manifests/shards using the current Node command surface, then verify load + inference. (project)
---

# DOPPLER Convert Skill

Use this skill to add or re-convert models for Doppler runtime.

## Execution Plane Contract

- JSON is the conversion contract (presets, manifests, converter config).
- JS is orchestration (parsing, conversion flow, validation, and artifact emission).
- WGSL is not selected here; compute policy is resolved later at runtime by manifest + kernel-path rules.
- Conversion must remain config-first and fail fast on unresolved kernel/policy requirements.

## Primary Conversion Commands

```bash
# Convert from Safetensors directory (or GGUF file path) via unified CLI
npm run convert -- --config '{
  "request": {
    "inputDir": "INPUT_PATH",
    "outputDir": "models/local/OUTPUT_ID",
    "convertPayload": {
      "converterConfig": {
        "output": {
          "modelBaseId": "OUTPUT_ID"
        }
      }
    }
  },
  "run": {
    "surface": "node"
  }
}'

# Same conversion through direct Node helper with converter-config JSON
node tools/convert-safetensors-node.js INPUT_PATH --config ./converter-config.json --output-dir models/local/OUTPUT_ID
```

Notes:
- `INPUT_PATH` can be a Safetensors directory, diffusion directory, or `.gguf` file.
- Unified CLI convert path is `tools/doppler-cli.js` -> `runNodeCommand()` -> `src/tooling/node-converter.js`.
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
npm run debug -- --config '{"request":{"modelId":"OUTPUT_ID","runtimePreset":"modes/debug"},"run":{"surface":"auto"}}' --json
```

For publication candidates, the verification bar is higher:

1. Promote successful ad hoc configs
- If the conversion used a temporary or inline config and the model runs successfully, copy/promote that config into `tools/configs/conversion/` so the conversion is reproducible.

2. Run an actual coherence check
- Use a deterministic prompt and deterministic sampling, not just a load-only run.
- Recommended shape:

```bash
npm run debug -- \
  --config '{"request":{"modelId":"OUTPUT_ID","runtimePreset":"modes/debug"},"run":{"surface":"auto"}}' \
  --runtime-config '{"shared":{"tooling":{"intent":"verify"}},"inference":{"prompt":"Explain what this model is in one short sentence.","sampling":{"temperature":0,"topK":1}}}' \
  --json
```

- Inspect `result.output` (and summary metrics) for non-empty, coherent text.

3. Pause for HITL review before promotion
- Summarize the prompt and observed output for the human.
- Before adding `models/catalog.json` entries, syncing support-matrix metadata, or uploading/publishing to Hugging Face, stop and ask for confirmation.

4. Offer optional perf validation
- If the output looks correct, propose:
  - `npm run bench -- --config ... --json`
  - `node tools/vendor-bench.js ...`
  - `node tools/compare-engines.js ...`

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
- `src/tooling/node-converter.js`
- `src/converter/core.js`
- `src/converter/conversion-plan.js`
- `docs/formats.md`
- `AGENTS.md`

## Related Skills

- `doppler-debug` for runtime correctness after conversion
- `doppler-bench` for perf regressions between variants
