# Conversion Protocol

Referenced by: `doppler-convert`, `doppler-debug`

## Conversion Triage Protocol (Required)

When a freshly converted model regresses, separate conversion integrity from runtime regressions before changing the conversion config:

1. Verify source dtypes from checkpoint headers (`BF16`/`F16`/`F32` mix).
2. Verify converted manifest fields: `quantization`, `quantizationInfo`, `inference.execution`.
3. Verify shard integrity (sampled shard hashes must match manifest hashes).
4. Verify numeric sanity by sampling tensor values from source vs converted bytes.
5. Verify parsed layer pattern semantics from manifest (Gemma `every_n` is layer 0 + every N).

Do not claim a conversion bug unless steps 1-4 fail.
Do not claim a runtime bug unless steps 1-4 pass and runtime still diverges.

## Conversion Promotion Gate (Required)

When a conversion is intended for reuse, registry inclusion, or Hugging Face publication:

1. Keep the conversion reproducible
- If the model was converted with an ad hoc or temporary config and the local run succeeds, promote that config into `src/config/conversion/` before treating the workflow as reusable.

2. Prove the model produces coherent output
- Do not stop at manifest/shard validation or load success.
- Run a real inference/debug pass with a deterministic prompt via runtime config (for example `runtime.inference.prompt` plus deterministic sampling) and inspect the emitted text in the command result/report.
- Treat empty, collapsed, NaN-like, or obviously incoherent output as a failed candidate even if the command exits successfully.

3. Put a human in the loop before publication
- Summarize the exact prompt used and the observed output for the user.
- Before adding or updating `models/catalog.json`, syncing support-matrix/catalog metadata, or uploading/publishing artifacts to Hugging Face, stop and ask the human to review the coherence result and confirm whether to proceed.

4. Offer performance follow-up before publication
- When a candidate looks correct, propose optional perf validation before registry/HF promotion:
  - `npm run bench`
  - vendor benchmark / compare-engine runs (`node tools/vendor-bench.js ...`, `node tools/compare-engines.js ...`)
- If catalog entries change after approval, update derived docs with `npm run support:matrix:sync`.
