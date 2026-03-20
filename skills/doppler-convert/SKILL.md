---
name: doppler-convert
description: Convert GGUF or SafeTensors assets into Doppler RDRR manifests/shards using the current Node command surface, then verify load + inference. (project)
---

# DOPPLER Convert Skill

Use this skill to add or re-convert models for Doppler runtime.

## Mandatory Style Guides

Read these before non-trivial conversion or manifest-contract changes:
- `docs/style/general-style-guide.md`
- `docs/style/javascript-style-guide.md`
- `docs/style/config-style-guide.md`
- `docs/style/command-interface-design-guide.md`

## Developer Guide Routing

For additive or extension-oriented conversion work, also open:
- `docs/developer-guides/README.md`

Then route to the matching playbook:
- new checked-in conversion recipe: `docs/developer-guides/04-conversion-config.md`
- model-family onboarding needed before conversion works: `docs/developer-guides/04-conversion-config.md` or `docs/developer-guides/composite-model-family.md`
- publication or curated metadata work: `docs/developer-guides/05-promote-model-artifact.md`
- new quantization/runtime artifact format: `docs/developer-guides/14-quantization-format.md`

## Execution Plane Contract

- JSON is the conversion contract (config assets, manifests, converter config).
- JS is orchestration (parsing, conversion flow, validation, and artifact emission).
- WGSL is not selected here; compute policy is resolved later at runtime by manifest + kernel-path rules.
- Conversion must remain config-first and fail fast on unresolved kernel/policy requirements.

## Primary Conversion Commands

```bash
# Convert from Safetensors directory (or GGUF file path) via unified CLI
# Output to a temporary directory first, then copy to external volume after verification.
npm run convert -- --config '{
  "request": {
    "inputDir": "INPUT_PATH",
    "outputDir": "/tmp/OUTPUT_ID-rebuild",
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
node tools/convert-safetensors-node.js INPUT_PATH --config ./converter-config.json --output-dir /tmp/OUTPUT_ID-rebuild
```

Notes:
- `INPUT_PATH` can be a Safetensors directory, diffusion directory, or `.gguf` file.
- Unified CLI convert path is `src/cli/doppler-cli.js` -> `runNodeCommand()` -> `src/tooling/node-converter.js`.
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
test -f /tmp/OUTPUT_ID-rebuild/manifest.json

# 2) Verify key manifest fields
jq '.modelId, .modelType, .quantization, .quantizationInfo, .inference.defaultKernelPath' /tmp/OUTPUT_ID-rebuild/manifest.json

# 3) Verify shards exist
ls /tmp/OUTPUT_ID-rebuild/shard_*.bin | wc -l

# 4) After verification passes, copy to external volume (source of truth)
cp -a /tmp/OUTPUT_ID-rebuild/. /media/x/models/rdrr/OUTPUT_ID/

# 4) Sanity-run inference
npm run debug -- --config '{"request":{"modelId":"OUTPUT_ID","runtimeProfile":"profiles/verbose-trace"},"run":{"surface":"auto"}}' --json
```

For publication candidates, the verification bar is higher:

1. Promote successful ad hoc configs
- If the conversion used a temporary or inline config and the model runs successfully, copy/promote that config into `src/config/conversion/` so the conversion is reproducible.

2. Run an actual coherence check
- Use a deterministic prompt and deterministic sampling, not just a load-only run.
- Recommended shape:

```bash
npm run debug -- \
  --config '{"request":{"modelId":"OUTPUT_ID","runtimeProfile":"profiles/verbose-trace"},"run":{"surface":"auto"}}' \
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

- `src/cli/doppler-cli.js`
- `tools/convert-safetensors-node.js`
- `src/tooling/node-command-runner.js`
- `src/tooling/node-converter.js`
- `src/converter/core.js`
- `src/converter/conversion-plan.js`
- `docs/rdrr-format.md`
- `docs/developer-guides/README.md`
- `AGENTS.md`

## Related Skills

- `doppler-debug` for runtime correctness after conversion
- `doppler-bench` for perf regressions between variants
