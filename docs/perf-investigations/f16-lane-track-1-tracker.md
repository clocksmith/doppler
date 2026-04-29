# Track 1: Doppler F16 Evidence

## Scope

Track 1 owns Doppler-side f16 lanes and frozen reference capture for:

- `gemma-4-31b-it-text-q4k-ehf16-af16`
- Qwen 3.6 27B f16 sibling, using the existing Qwen 3.6 Q4K weight pack

Existing f32 lanes remain guardrails:

- `gemma-4-31b-it-text-q4k-ehf16-af32`
- `qwen-3-6-27b-q4k-ehaf16` with `variantTag=q4k-ef16-af32`

Track 1 does not change Doe/CSL lowering or existing f32 receipts.

## Status

| Item | Status | Notes |
| --- | --- | --- |
| Gemma 4 31B af16 manifest sibling | Done | Existing `weightsRef` manifest at `models/local/gemma-4-31b-it-text-q4k-ehf16-af16/manifest.json`. |
| Gemma 4 31B af16 runtime profile | Done | Existing profile `profiles/gemma4-31b-f16-activations-probe`. |
| Gemma 4 31B af16 transform | Done | Existing `useGemma431BTextF16Activations` transform and tests route the f16 lane. |
| Gemma 4 31B af16 frozen reference | Done | Browser/WebGPU and Node/WebGPU both decode `blue`; Doe fixture `r3-1-31b-doppler-frozen-af16` validates with `q4k-ehf16-af16`. |
| Qwen 3.6 af32 guardrail | Done | Existing `qwen-3-6-27b-q4k-ehaf16` is the f32 lane despite the legacy model id. |
| Qwen 3.6 af16 manifest sibling | Done | Added `models/local/qwen-3-6-27b-q4k-eaf16/manifest.json` as a `weightsRef` sibling with f16 compute/session policy. |
| Qwen 3.6 af16 runtime profile | Done | Added experimental profile `profiles/qwen3-6-27b-f16-activations-probe`. |
| Qwen 3.6 af16 transform | Done | Added `useQwen36F16Activations` and a capability rule that routes the f16 lane onto f16 kernels. |
| Reference-run `weightsRef` preflight | Done | `run-program-bundle-reference.js` checks shared weight-pack files for manifest-only siblings instead of requiring local shard copies. |
| Fixture-capture command docs | Done | `docs/cerebras-tsir-fixture-capture.md` has f32 and f16 commands with explicit output/report paths and Qwen L=3 capture. |
| Qwen 3.6 af16 frozen reference | Done | Node/WebGPU decodes coherent text ending in `blue`; Doe fixture `r3-2-27b-doppler-frozen-af16` validates with `q4k-eaf16`. |
| Qwen 3.6 af16 browser coherence | Follow-up | Browser attempt did not produce a report before manual stop; Node/WebGPU fixture and validator are bound. |

## Guardrails

- Existing af32 manifests, fixture paths, and receipts are not overwritten.
- `variantTag` is display metadata; dtype truth comes from `quantizationInfo` and session dtype fields.
- Qwen af16 must share the existing Qwen weight pack through `weightsRef`; it must not duplicate shards.
- Any af32 drift investigation is a separate artifact-only track until a reproducible command and receipt exist.

## Open Contract Item

The current Qwen 3.6 af32 manifest does not declare `artifactIdentity.shardSetHash`.
The af16 sibling pins the legacy target `artifactIdentity.weightPackHash` as the
shared shard identity, and runtime `weightsRef` validation accepts that target
identity when `shardSetHash` is absent. The af16 frozen reference now carries
and validates its explicit dtype profile.
