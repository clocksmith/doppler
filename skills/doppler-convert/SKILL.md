---
name: doppler-convert
description: Convert identified GGUF or SafeTensors assets into Doppler RDRR manifests and shards, then verify load and inference for the specified model.
---

# DOPPLER Convert Skill

## Prerequisites

Supply the source model path and format, immutable source identity, intended Doppler
model ID, conversion configuration, output root, and the inference probe to validate.

## Procedure

1. Read the required conversion contracts and select the matching documented command.
2. Dry-run or inspect the resolved plan, then convert into the identified output root.
3. Validate manifest and shard integrity, load the exact generated model, and run the
   supplied inference probe before any copy or promotion.

## Style Guides

Before non-trivial conversion or manifest-contract changes, open the guides in the
Conversion row of the [AGENTS.md owner table](../../AGENTS.md#style-guides).

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
jq '.modelId, .modelType, .quantization, .quantizationInfo, .inference.schema, (.inference.session != null), (.inference.execution != null)' /tmp/OUTPUT_ID-rebuild/manifest.json

# 3) Verify shards exist
ls /tmp/OUTPUT_ID-rebuild/shard_*.bin | wc -l

# 4) After verification passes, copy to the external RDRR root (source of truth):
#    $DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/OUTPUT_ID/ (see models/GUIDE.md)
: "${DOPPLER_EXTERNAL_MODELS_ROOT:?set DOPPLER_EXTERNAL_MODELS_ROOT first (e.g. /Volumes/models)}"
mkdir -p "$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/OUTPUT_ID"
cp -a /tmp/OUTPUT_ID-rebuild/. "$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/OUTPUT_ID/"

# 5) Sanity-run inference
npm run cli -- profiles --json
npm run debug -- --config '{"request":{"modelId":"OUTPUT_ID"},"run":{"surface":"auto"}}' --runtime-profile profiles/verbose-trace --json
```

Notes:
- For execution-v1 artifacts, `inference.schema`, `inference.session`, and `inference.execution` are the important contract checks.
- Do not use `inference.defaultKernelPath` as a required success criterion for v1 manifests.

## Conversion Format vs Benchmark Parity

Conversion success is not benchmark parity. RDRR, GGUF, ONNX, SafeTensors, and
LiteRT artifacts can differ in tokenizer packaging, quantization layout, dtype
materialization, chat template handling, and runtime cache behavior. Treat
format differences as disclosed benchmark metadata, not as proof that two
engines are doing identical work.

Before using a converted artifact in benchmark claims, hand off to
`doppler-bench` and require the relevant fairness gates: artifact identity,
format disclosure, runtime surface, fallback status, timing scope, work
accounting, and correctness policy.

For publication candidates, follow the
[Conversion Promotion Gate](../../docs/agents/conversion-protocol.md#conversion-promotion-gate-required)
(promote the config, prove coherent output, stop for human review, offer perf
follow-up). The coherence check from this skill:

```bash
npm run debug -- \
  --config '{"request":{"modelId":"OUTPUT_ID"},"run":{"surface":"auto"}}' \
  --runtime-config '{"inference":{"prompt":"Explain what this model is in one short sentence.","sampling":{"temperature":0,"topK":1}}}' \
  --json
```

Inspect `result.output` (and summary metrics) for non-empty, coherent text.

## Conversion Triage Contract

When conversion quality is in question, follow the
[Conversion Triage Protocol](../../docs/agents/conversion-protocol.md#conversion-triage-protocol-required).

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

## Validation

The generated manifest and shards pass load validation and complete the documented
inference probe with the expected model identity and output contract.

## Stop Conditions

Stop before copying or promoting artifacts when source identity, conversion
configuration, shard integrity, or inference verification is missing. Never infer
parity from conversion success alone.

## Outputs

An RDRR manifest and shards bound to the source/config identities, plus load and
inference evidence for the specified model.

## Side Effects

Writes conversion artifacts to the supplied output root and may load them locally.
Copying, registry changes, promotion, and benchmark claims require separate requests.
