# Migrating from DopplerX 0.1.7 to 0.2.0

DopplerX 0.2.0 is a breaking release for one reason: every enum value Doppler
owns is now spelled in kebab-case. Nothing else in the public API, the Capsule
formats or the defaults changed between 0.1.7 and 0.2.0. Error codes
(`DOPPLER_*`), export names, function signatures, object keys and schema ids
are unchanged.

## What a consumer must change

Doppler now **emits** only kebab-case. Code that compares a value Doppler
returns, or narrows on its TypeScript type, must use the new spelling.

| Where the value appears | 0.1.7 | 0.2.0 |
| --- | --- | --- |
| Provider `failureClass`, receipts, `fallbackDecision.reason` | `gpu_device_lost`, `gpu_oom`, `gpu_timeout`, `gpu_unsupported`, `gpu_unavailable`, `model_load_failed`, `policy_denied`, `runtime_internal`, `fallback_failed` | `gpu-device-lost`, `gpu-oom`, `gpu-timeout`, `gpu-unsupported`, `gpu-unavailable`, `model-load-failed`, `policy-denied`, `runtime-internal`, `fallback-failed` |
| Provider fallback surface and policy reasons | `openai_compat`, `policy_local_only`, `policy_cloud_only`, `policy_prefer_cloud`, `no_fallback_configured` | `openai-compat`, `policy-local-only`, `policy-cloud-only`, `policy-prefer-cloud`, `no-fallback-configured` |
| `encodeSequence()` result `alphabet`, manifest `inference.sequence.alphabet`, `sequenceAlphabet` | `amino_acid` | `amino-acid` (`nucleotide` is unchanged) |
| Sequence qualification abort (`sequenceQualificationAbort`) | `before_execution`, `after_start` | `before-execution`, `after-start` |
| Structured-head `modelType` | `structured_json_head`, `dream_structured`, `dream_intent_posterior_head`, `dream_d1_to2_bridge`, `dream_synthesis`, `dream_energy_compose`, `energy_row_head`, `dream_energy_head` | the same words joined with `-` |
| Diffusion scheduler `type` | `flowmatch_euler`, `euler_a`, `dpmpp_2m` | `flowmatch-euler`, `euler-a`, `dpmpp-2m` |
| KV cache `layout`, tiering and quant modes, gating | `contiguous_quantized`, `bdpa_paged`, `turboquant_prod`, `force_on`, `force_off` | `contiguous-quantized`, `bdpa-paged`, `turboquant-prod`, `force-on`, `force-off` |
| Tooling `diagnostics`, `receiptPolicy` | `on_failure` | `on-failure` |
| Decode stats `decodeMode`, `decodeStrategy` | `single_token`, `batched_gpu`, `batched_gpu_stepwise_ple`, `self_speculation`, `replay_prefill` | `single-token`, `batched-gpu`, `batched-gpu-stepwise-ple`, `self-speculation`, `replay-prefill` |
| Execution plan id | `finiteness_fallback` | `finiteness-fallback` |
| Shard loader `lastSource.fallback` (output only) | `custom_range_unavailable`, `custom_range_not_supported`, `custom_stream_not_supported[_resume]`, `custom_stream_interrupted[_resume]`, `custom_stream_partial_resume`, `custom_range_partial_retry` | the same words joined with `-` |
| Manifest per-layer inputs, dtype transitions, layer patterns, FFN branch mode, INT4 encodings, lowering classes, training stages and objectives, bench lanes, precision replay modes, operator events, support status | snake_case | kebab-case; the complete list is the table in [`src/config/legacy-enum-values.js`](../../src/config/legacy-enum-values.js) |
| Trainer handoff format | `huggingface_safetensors` | `huggingface-safetensors` |
| Translation artifact lane | `selective_f16` | `selective-f16` |

The TypeScript declarations changed the same way: for example `FailureClass`,
`KVLayout`, `KVTieringMode`, `ToolingDiagnosticsMode`, `DiffusionSchedulerType`
and `StructuredJsonHeadModelType` now list only the kebab-case values, so a
comparison against an old literal fails to type-check.

Values that did **not** change:

- Execution-graph op ids and probe stage names (`q_proj`, `input_norm`,
  `logits_final`). They mirror Hugging Face module names and are bound into
  published manifests and `executionGraphHash` digests.
- The decode batch-guard reasons in decode stats other than
  `multi-token-decode-disabled` (for example `batch_size_1`, `no_gpu`,
  `bdpa_paged_layout`, `linear_attention_layers`). They are diagnostic labels and
  keep their 0.1.7 spelling in 0.2.0.
- Third-party names Doppler passes through (Hugging Face, tokenizer and vendor
  identifiers).

## Supported compatibility reads

Doppler still **reads** the old spellings. Every reader below maps a legacy
value to its kebab-case form before validation, so the code behind it only sees
the new spelling. Unknown values are not rewritten; the owning validator still
rejects them.

| Input | Reader | Kept |
| --- | --- | --- |
| RDRR manifests (`manifest.json`), including published, digest-bound ones | `parseManifest` (`src/formats/rdrr/parsing.js`) | as long as manifests written before 0.2.0 are supported; their bytes and hashes are not rewritten |
| Runtime config overrides: `loadOptions.runtimeConfig`, `setRuntimeConfig`, `resolveRuntimeConfig`, runtime profiles and every runtime-config merge | `resolveRuntimeConfig` and `mergeRuntimeValues` (`src/config/runtime.js`, `src/config/runtime-merge.js`) | through the 0.x releases |
| Provider `policy.fallbackOn` | `normalizeFailureClass` (`src/client/failure-taxonomy.js`); the type accepts `LegacyFailureClass` | through the 0.x releases |
| Pipeline `modelType` for structured heads | `normalizeModelType` (`src/inference/pipelines/registry.js`) | through the 0.x releases |
| Diffusion scheduler `type` | `assertSupportedDiffusionSchedulerType` and the diffusion init path | through the 0.x releases |
| Command requests: training stage | `src/tooling/command-request-normalization.js` | through the 0.x releases |
| Training workload files (`distill.studentGraphMode`, `stagePlan[].id`) | `src/experimental/training/workloads.js` | through the 0.x releases |
| Trainer handoff descriptors (`artifact.format`, including Gamma's `huggingface_safetensors`) | `src/experimental/bridge/trainer-artifact-bridge.js` | through the 0.x releases |
| The recorded 2026-07-13 translation readiness receipt | `readTranslationArtifactCompetitionReceipt` verifies it with its original spelling and hash and returns kebab-case lanes | indefinitely; the receipt is historical evidence |

"Through the 0.x releases" means a later 0.x minor release may remove a
reader only with a CHANGELOG entry one minor release ahead. Manifest reads stay
until Doppler stops loading pre-0.2.0 manifests at all. The alias table is
closed: no new snake_case value is ever added to it
(`tools/check-kebab-enum-values.js`).

Reads are not writes. Doppler never echoes a legacy spelling back: a receipt,
stats object or result produced from a legacy input carries the kebab-case
value. A consumer that compares Doppler output against values it stored itself
before 0.2.0 must map its own stored values.

Tests: `tests/integration/legacy-enum-reads.test.js` (runtime config, manifest,
provider, pipeline and scheduler reads), `tests/integration/failure-taxonomy.test.js`
and `tests/tooling/check-kebab-enum-values.test.js`.
