# Inference Debug Protocol

Referenced by: `doppler-debug`

When a model loads but produces incoherent output, follow this fail-closed debug ladder.
Do not skip ahead to architecture theories or benchmark tweaks.

## 0. Check manifest-config parity first

- Read the manifest on disk and the conversion config in `src/config/conversion/`.
- If any inference field disagrees (dtype, kernel, session, layerPattern), the manifest is stale — re-refresh it, do not patch runtime code.
- The conversion config is the source of truth. The manifest is a stamped artifact.

## 1. Classify the failure

- `tokenization / chat-template`
- `conversion / artifact integrity`
- `runtime numerics`
- `surface / harness parity`
- `benchmark-only`

## 2. Establish one trusted reference before changing code

- For model-quality failures, get a deterministic reference from the source runtime when possible.
- Capture: exact prompt text, exact token IDs, one early activation slice, one output/logits slice.

## 3. Use boundary diffs, not broad speculation

Compare this sequence and stop at the first divergent boundary:
- embeddings
- post input norm
- Q/K/V pre-RoPE
- Q/K post-RoPE
- attention output
- FFN output
- final logits

## 4. Quantized failure control

Run one F16 or source-precision control before touching quantized kernels:
- F16/source-precision good + quantized bad => quantized path issue
- F16/source-precision bad + quantized bad => shared conversion/layout/runtime issue

## 5. Stop prompt/harness churn once token IDs match

If token IDs or embeddings already match, do not keep changing templates, harnesses, or benchmark wrappers until a later boundary proves they are relevant.

## 6. Prefer one new probe over one new theory

Add the smallest permanent/config-driven probe needed to classify the next boundary.
Do not add throwaway logs.

## 7. Conversion status is fail-closed

Do not mark a conversion as complete unless all of these exist and agree:
- successful process exit
- `manifest.json`
- expected shard set
- valid conversion report

A directory with shards but no manifest is an interrupted conversion, not a usable artifact.

Canonical workflow doc: `docs/debug-playbook.md`
Reusable report template: `docs/debug-investigation-template.md`
