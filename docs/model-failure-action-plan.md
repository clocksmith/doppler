# Model Failure Action Plan

Last updated: 2026-03-12T19:14:37Z
Plan status: active
Current resume point: `WS4.6`
Current highest-priority ready step: `WS4.6`

## Purpose

This is the canonical action plan for investigating and fixing the current Doppler model-failure cluster.

Primary scope:

- `gemma-3-270m-it-q4k-ehf16-af32`
- `gemma-3-1b-it-q4k-ehf16-af32`
- `translategemma-4b-it-q4k-ehf16-af32`
- `qwen-3-5-0-8b-q4k-ehaf16`
- `qwen-3-5-2b-q4k-ehaf16`

Secondary maintenance scope:

- sampled stale local artifacts such as `gemma-3-1b-it-f16-af32`
- `gemma2` execution-v0 graph regression
- `lfm2` runtime-path verification and fallback audit
- catalog/support-matrix metadata cleanup

This file is intentionally explicit, resumable, and hypothesis-aware. Do not replace it with ad hoc notes. Update it as work progresses.

## How To Use This File

1. Read the status legend, evidence rules, and current ground truth before touching code.
2. Resume the first step marked by `Current resume point`, or the earliest unfinished step in the highest-priority workstream whose entry gate is satisfied.
3. When a step changes state, update:
   - the step checkbox
   - the workstream status
   - the current resume point near the top of this file
   - the progress log at the end of this file
4. When a hypothesis is disproved, do not delete it. Mark it `disproved` and record the evidence.
5. Do not treat a model as fixed until there is real deterministic runtime evidence, not only schema or harness evidence.
6. Before updating `models/catalog.json`, syncing the support matrix, or making any release-facing claim, stop and get human confirmation on the observed output.

## Status Legend

| Status | Meaning |
| --- | --- |
| `not_started` | No work has been done yet. |
| `ready` | Safe to pick up now; entry gate is satisfied. |
| `in_progress` | Actively being worked. Only one workstream should usually be in this state. |
| `blocked` | Cannot proceed until the named blocker is cleared. |
| `needs_recheck` | New evidence invalidated a prior assumption; re-verify before continuing. |
| `validated` | The fix or finding is supported by acceptable evidence, but follow-up or human review is still required. |
| `done` | Completed, reviewed, and no follow-up is needed. |
| `disproved` | The hypothesis or path was checked and rejected by evidence. |

## Evidence Rules

### Evidence Classes

| Class | What it means | Can it close correctness work? |
| --- | --- | --- |
| `A` | Real deterministic runtime evidence with actual device/provider info, real artifact loading, real prompt/output, and matching config/manifest context | Yes |
| `B` | Contract evidence: manifest inspection, schema validation, conversion logs, slot-graph checks, targeted tests, preset/config diffs | Only contract/config items |
| `C` | Synthetic harness or fixture output, canned generations, null device info, fixed `1 ms`/`1000 tok/s`, mock outputs | No |

### Non-Negotiable Rules

1. Any report artifact with null or missing device/provider info, synthetic-looking fixed timings such as exactly `1 ms` durations or exactly `1000 tok/s` throughput, or canned/mock generation output is Class `C` regardless of its timestamp. The `reports/**/2026-03-12T13-48*` and `2026-03-12T13-49*` batches are known Class `C` examples.
2. Do not claim a conversion bug until the conversion triage protocol passes through source dtypes, manifest fields, shard integrity, and sampled numeric sanity.
3. Do not claim a runtime bug until conversion/artifact integrity is established first.
4. Do not invent runtime defaults or hidden fallbacks. If behavior changes, it must come from manifest, preset, runtime config, or rule assets.
5. `null` is valid only when the contract explicitly allows an explicit disable. Missing required fields are not acceptable.
6. Metadata issues such as `baseUrl` or support-matrix drift are real problems, but they are not by themselves proof of model-runtime failure.

## Current Ground Truth

Repo-backed facts as of 2026-03-12:

1. The support matrix currently marks:
   - `gemma-3-270m-it-q4k-ehf16-af32` as failing on 2026-03-11
   - `gemma-3-1b-it-q4k-ehf16-af32` as failing on 2026-03-11
   - `qwen-3-5-0-8b-q4k-ehaf16` as failing on 2026-03-06 with linear-attention correctness still unverified
   - `qwen-3-5-2b-q4k-ehaf16` as failing on 2026-03-06 for the same family reason
   - `translategemma-4b-it-q4k-ehf16-af32` as verified on browser on 2026-03-06
2. `translategemma-4b-it-q4k-ehf16-af32` has an explicit `sessionDefaults.kvcache.layout: "paged"` in its checked-in conversion config.
2a. Git history shows the checked-in TranslateGemma conversion config first appears on 2026-03-08, after the recorded 2026-03-06 browser verification. That means the current checked-in `layout: "paged"` policy postdates the recorded verification and `WS2` should treat the contiguous-layout path as a re-verification of current policy rather than the exact previously verified artifact.
3. [`src/inference/pipelines/text/init.js`](../src/inference/pipelines/text/init.js) only blocks threshold-based `contiguous -> paged` auto-upgrade when `forceContiguousKVCache` is true. It does not rewrite an explicit `paged` request back to `contiguous`.
4. The sampled local report artifacts from 2026-03-12 are mostly Class `C` evidence. They are useful for contract artifacts, but not for generation quality or performance claims. This includes the `2026-03-12T13-48*` and `2026-03-12T13-49*` batches and later reports with the same null-device and fixed-`1 ms`/`1000 tok/s` signature, such as [`reports/translategemma-4b-it-q4k-ehf16-af32/2026-03-12T15-34-44.706Z.json`](../reports/translategemma-4b-it-q4k-ehf16-af32/2026-03-12T15-34-44.706Z.json) and [`reports/qwen-3-5-0-8b-q4k-ehaf16/2026-03-12T15-34-41.292Z.json`](../reports/qwen-3-5-0-8b-q4k-ehaf16/2026-03-12T15-34-41.292Z.json).
5. The earlier "Gemma F32a path is the prime suspect because TranslateGemma uses F16a" story is no longer a lead explanation. Kernel-level review reported no arithmetic divergence between the paired F32a and F16a WGSL paths beyond I/O and storage dtype differences, so any remaining F32a vs F16a comparison belongs at the orchestration layer rather than kernel math.
6. There is real Class `A` evidence that `gemma-3-1b-it-q4k-ehf16-af32` produced coherent output on at least one prompt on 2026-03-11 in [`reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json`](../reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json). This contradicts the support-matrix "failing" summary and means the 1B Q4K path must not be treated as categorically broken.
7. Qwen remains the strongest repo-backed case for a family-specific runtime correctness issue centered on linear attention and Qwen-specific semantics.
8. The exact target-ID local artifacts for `qwen-3-5-0-8b-q4k-ehaf16`, `qwen-3-5-2b-q4k-ehaf16`, `gemma-3-270m-it-q4k-ehf16-af32`, and `gemma-3-1b-it-q4k-ehf16-af32` have now been refreshed from source on 2026-03-12. The refreshed Qwen artifacts no longer fail required-field checks, but they still truthfully record `schema: null`, `defaultKernelPath: null`, and no inline execution graph. The refreshed Gemma artifacts retain `doppler.execution/v0`, `gemma3-q4k-dequant-f32a-online`, and passing execution-v0 graph checks.
9. Refreshed Qwen artifacts currently take the legacy text-pipeline path in node debug and test-model flows, not execution-v0. `defaultKernelPath: null` is intentional and allowed for this family, but it leaves the model without a model-level kernel-path anchor; linear-attention layers still route into `runLinearAttentionLayer()` by `layerPattern`, full-attention layers still dispatch the standard attention kernels, and lower-level matmul kernels auto-select generically from capability-based selectors.
10. Real Class `A` node/WebGPU debug runs on Apple M3 from 2026-03-12 show that refreshed `qwen-3-5-0-8b-q4k-ehaf16` and the local `qwen-3-5-0-8b-f16` comparison artifact both load successfully and both produce incoherent output. The exact Q4K artifact emits newline/period spam in [`reports/qwen-3-5-0-8b-q4k-ehaf16/2026-03-12T16-45-00.422Z.json`](../reports/qwen-3-5-0-8b-q4k-ehaf16/2026-03-12T16-45-00.422Z.json), while the F16 comparison artifact emits multilingual gibberish in [`reports/qwen-3-5-0-8b-f16/2026-03-12T16-47-51.839Z.json`](../reports/qwen-3-5-0-8b-f16/2026-03-12T16-47-51.839Z.json). This weakens any Q4K-only explanation for the core Qwen incoherence.
11. Source checkpoint indexes for both Qwen 3.5 0.8B and 2B expose learned `self_attn.q_norm` and `self_attn.k_norm` weights only on the six full-attention layers, while the linear-attention layers expose `linear_attn.in_proj_z`, `in_proj_a`, and `in_proj_b` weights instead. That makes extra learned Q/K normalization on linear-attention layers unlikely and keeps the focus on delta-net semantics, linear-state math, and integration correctness.
12. Real Class `A` node/WebGPU debug runs on Apple M3 from 2026-03-12 show that the refreshed exact Gemma Q4K artifacts do not fail uniformly: [`reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-12T17-09-40.767Z.json`](../reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-12T17-09-40.767Z.json) produces a coherent “Blue” answer on the exact execution-v0 path, while [`reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-09-40.764Z.json`](../reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-09-40.764Z.json) still produces incoherent counting text on the same prompt. Combined with matching manifest hash checks and exact sampled Q4K block parity against source BF16 weights, this points away from a generic Gemma Q4K conversion failure and toward a model-specific runtime or numeric-sensitivity issue.
13. A controlled 2026-03-12 Apple M3 node/WebGPU run that explicitly overrides `gemma-3-270m-it-q4k-ehf16-af32` to `gemma3-q4k-dequant-f32a-nosubgroups` still produces incoherent output in [`reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-12-30.280Z.json`](../reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-12-30.280Z.json). The override also fails closed unless the runtime pins matching `f32` activation/output dtypes, so subgroup-only explanations are now weaker while orchestration-layer dtype handling remains relevant.
14. Additional Class `A` Apple M3 node/WebGPU probe runs from 2026-03-12 now show that `gemma-3-270m-it-q4k-ehf16-af32` stays numerically identical through the audited early runtime path whether it uses the manifest-inline execution-v0 kernel path or an explicit `runtime.inference.kernelPath: "gemma3-q4k-dequant-f32a-nosubgroups"` override with pinned matching dtypes. The one-token layer-0 probe runs in [`reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-29-12.222Z.json`](../reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-29-12.222Z.json) and [`reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-29-12.224Z.json`](../reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-29-12.224Z.json) match at `embed_out`, layer-0 `q_proj`, `attn_out`, `ffn_out`, `layer_out`, final norm, and sampled first-token logits to the logged precision, and a three-token deterministic decode in [`reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-31-26.778Z.json`](../reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-31-26.778Z.json) and [`reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-31-26.776Z.json`](../reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-31-26.776Z.json) produces the same `"\nDeterminate"` prefix with matching layer-0 decode probes. This materially weakens kernel-path-selection and subgroup-only explanations for the 270M failure.
15. A directional Class `A` local comparison run against [`reports/gemma-3-270m-it-f16-af32/2026-03-12T17-29-12.234Z.json`](../reports/gemma-3-270m-it-f16-af32/2026-03-12T17-29-12.234Z.json) shows that the local 270M F16 comparison artifact already differs from the refreshed Q4K artifact at `embed_out` and diverges further by layer-0 `q_proj` and first-token logits. Because this F16 artifact is not one of the WS4 freshness-gated targets, treat it as directional evidence for “divergence begins no later than prefill/layer 0” rather than as closure of the root cause.
16. Contract work in `WS4.6` closed one silent dtype-propagation hole on 2026-03-12: [`src/gpu/weight-buffer.js`](../src/gpu/weight-buffer.js) now returns tagged runtime dtype metadata for raw `GPUBuffer` weights, the text embedding entrypoints now consult that accessor for raw buffers instead of wrapper-only metadata, and [`tests/inference/weight-buffer-runtime-dtype.test.js`](../tests/inference/weight-buffer-runtime-dtype.test.js) proves that QKV fusion preserves `f16` semantics for tagged raw buffers even when pooled buffer sizes would otherwise make them look like `f32` by size. This is Class `B` evidence only until the real 270M runtime smokes are re-run on the patched build.
17. A follow-up Class `A` Apple M3 node/WebGPU replay on the patched build, [`reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-57-13.308Z.json`](../reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-12T17-57-13.308Z.json), shows that the refreshed 270M runtime path was already loading embeddings as `WeightBuffer` (`isWeightBuffer=true` in the loader trace), still gathers them as `embeddingDtype=f16`, and reproduces the same probed `embed_out`, layer-0 `q_proj`, `attn_out`, `ffn_out`, `layer_out`, `final_norm`, and sampled first-token logits as the pre-fix run to logged precision. This upgrades item 16 from “possible active cause” to “latent contract hole now closed, but not the active cause of the current 270M divergence on refreshed artifacts.”
18. Contract work in `WS4.6` also closed a BF16 matmul-weight policy gap on 2026-03-12: [`src/loader/tensors/tensor-loader.js`](../src/loader/tensors/tensor-loader.js) now respects `runtime.inference.compute.keepF32Weights` for BF16-origin matmul weights instead of silently narrowing them to `f16` whenever `shader-f16` is available, and [`tests/loader/tensor-loader-cleanup.test.js`](../tests/loader/tensor-loader-cleanup.test.js) now covers both the `keepF32Weights=false -> f16` and `keepF32Weights=true -> f32` branches. This is Class `B` evidence until a real runtime comparison actually exercises that debug/compat path on an affected model.
19. An offline Class `A` / `B` reconstruction on 2026-03-12 now clears layer-0 `q_proj` runtime correctness as the active 270M failure branch. Using the real prompt tokenization (`19` prompt tokens, final token ID `24369`), the stored local 270M embedding row and `input_layernorm.weight` reproduce the layer-0 `attn_normed` input exactly, and CPU matmul from those reconstructed inputs matches both the direct-source BF16 `q_proj` samples and the refreshed exact-Q4K `q_proj` samples to tight tolerance. The source path matches at ~`4e-5` max absolute error on sampled dims, while the Q4K path matches at ~`7e-3` max absolute error; the larger `270M` layer-0 `q_proj` drift is therefore explained by the quantized weights themselves, not by a loader, kernel-path-selection, or first-projection runtime mismatch.
20. A paired offline Class `B` comparison on the same prompt now shows that the legitimate layer-0 `q_proj` drift introduced by exact Q4K is materially larger on `gemma-3-270m-it` than on `gemma-3-1b-it`. Replaying both local F16 and exact-Q4K artifacts through local tokenization, embedding gather, Gemma embedding scaling, RMSNorm, and CPU `q_proj` matmul yields mean absolute `q_proj` deltas of about `1.40` for `270M` versus about `0.31` for `1B`, with sampled-dim drift up to about `6.07` on `270M` versus about `0.27` on the same sampled dims for `1B`. This strengthens the “model-specific numeric sensitivity / later-stage amplification” branch over any generic Gemma Q4K-runtime-mismatch story.

## Hypothesis Register

| ID | Hypothesis | Status | Current read | Next action |
| --- | --- | --- | --- | --- |
| `H-GEMMA-Q4K` | Gemma 3 Q4K failures are in the Q4K artifact/runtime path, not generic Gemma 3 execution | `in_progress` | Conversion triage now passes on refreshed local artifacts: source checkpoints are BF16, sampled local shard hashes match, sampled `q_proj` Q4K blocks match the source-derived quantization bytes exactly, exact runtime smokes diverge by model (`1B` coherent, `270M` incoherent), offline reconstruction shows the 270M layer-0 `q_proj` runtime samples match the stored exact-Q4K weights rather than an unexpected runtime branch, and the legitimate `q_proj` drift induced by Q4K is materially larger on `270M` than on `1B` | Continue the runtime-path audit after `q_proj`, with focus on later-stage amplification, model-specific numeric sensitivity, and any downstream stage where legitimate Q4K drift becomes pathological on 270M |
| `H-GEMMA-F32A` | Gemma F32a vs F16a kernel divergence is the primary root cause | `disproved` | Kernel-level review reported paired WGSL paths with matching arithmetic structure and f32 accumulation; only storage and I/O dtype differ | N/A |
| `H-GEMMA-F32A-ORCH` | Gemma F32a vs F16a orchestration-layer dtype propagation or buffer handling difference causes divergence | `needs_recheck` | Narrower still: explicit `nosubgroups` overrides fail closed without pinned `f32` runtime dtypes, but once pinned they now match the manifest-inline execution-v0 path through audited layer-0 probes and early decode on 270M. The raw-buffer dtype propagation hole is fixed and regression-covered, a patched Class `A` replay shows it was not active on the refreshed 270M path, and the BF16 `keepF32Weights` policy gap is now closed for future debug/compat runs | Continue on remaining non-kernel-path drift: loader/runtime metadata that does not depend on raw-buffer tags, model-specific numeric sensitivity, and any still-unreproduced BF16-vs-F16 edge cases that affect refreshed artifacts under real Class `A` runs |
| `H-TG-PAGED` | Explicit `paged` KV layout bypasses the contiguous-only intent for Gemma 3 mixed-attention models | `validated` | Confirmed by current `init.js` control flow; `WS2.1` should codify this as the pre-fix baseline | Add fail-fast and then align config |
| `H-TG-BROKEN` | TranslateGemma is fundamentally broken in Doppler | `needs_recheck` | Not supported by current repo evidence | Treat as unproven until real runtime repro exists |
| `H-QWEN-LA` | Qwen 3.5 failure is driven by linear-attention correctness or Qwen-specific runtime semantics | `in_progress` | Strongest remaining explanation: refreshed exact Q4K and F16 artifacts both fail in real node/WebGPU smokes after taking the same legacy path, so stale-manifest and Q4K-only explanations are now weaker | Strengthen tests around the best remaining runtime-semantic candidates |
| `H-QWEN-STALE` | Sampled local Qwen artifacts are stale or manifest-incomplete relative to the current contract | `done` | Validated for the old local artifacts; refreshed exact-ID Qwen artifacts now satisfy required-field checks but still need runtime-path investigation | Keep old artifacts out of runtime conclusions and continue in `WS3` |
| `H-TG-META` | TranslateGemma local-loading/catalog metadata is inconsistent | `validated` | Known metadata issue | Fix catalog truthfully and re-sync support matrix if needed |
| `H-LFM2-FALLBACK` | LFM2 no-subgroups path silently falls back to the wrong kernel family | `validated` | Confirmed by `kernel-path.rules.json`: LFM2 falls back to `gemma3-q4k-dequant-f32a-nosubgroups` on non-subgroup devices and on finiteness fallback, while the registry lacks an LFM2 no-subgroups path | Create an LFM2-specific no-subgroups kernel path and verify conv-layer dispatch |

## Master Status Board

| Workstream | Status | Priority | Depends on | Exit gate |
| --- | --- | --- | --- | --- |
| `WS0` Evidence normalization and inventory | `done` | P0 | none | Target artifacts, evidence classes, and current failure claims are pinned |
| `WS1` Artifact freshness and manifest contract repair | `done` | P1 | `WS0` | Target artifacts are current enough for runtime investigation |
| `WS2` TranslateGemma hardening and metadata | `ready` | P1 | none | Explicit paged-layout hazard is fail-closed and config/catalog are aligned |
| `WS3` Qwen runtime-path and linear-attention investigation | `validated` | P1 | `WS1` | Real runtime behavior is explained and validated with real evidence |
| `WS4` Gemma Q4K conversion and numeric triage | `in_progress` | P1 | `WS1` | Conversion/runtime split is resolved with real evidence |
| `WS5` Maintenance appendix: Gemma 1B F16, Gemma2, LFM2, catalog cleanup | `ready` | P2 | none | Side issues are either fixed, parked with evidence, or explicitly deprioritized |

## Order Of Operations

1. Finish `WS0` before drawing any new runtime conclusion.
2. Finish the artifact-freshness gate in `WS1` before closing anything in `WS3` or `WS4`.
3. `WS2` may run in parallel because the TranslateGemma paged-layout guard issue is a concrete code-path problem regardless of whether a fresh runtime repro exists today.
4. `WS5` must not block `WS2`, `WS3`, or `WS4`. Treat it as a maintenance appendix unless it produces new evidence that changes the primary workstreams.
5. Do not update release-facing metadata until a human has reviewed coherent output from a real deterministic smoke.

## WS0: Evidence Normalization And Inventory

Status: `done`

Goal: build one canonical inventory of target artifacts, evidence class, and next action so later work is anchored to real state rather than mixed reports.

Entry gate: none

Exit gate:

- every target model has a row in the inventory table
- every relevant report batch is tagged as Class `A`, `B`, or `C`
- every target model has exactly one current next action

Key files:

- [`docs/model-support-matrix.md`](./model-support-matrix.md)
- [`models/catalog.json`](../models/catalog.json)
- `models/local/**/manifest.json`
- `reports/**`

Target inventory template:

| Model ID | Local artifact path | Latest repo claim | Latest real runtime evidence | Latest contract-only evidence | Current next action |
| --- | --- | --- | --- | --- | --- |
| `gemma-3-270m-it-q4k-ehf16-af32` | refreshed exact local artifact: `models/local/gemma-3-270m-it-q4k-ehf16-af32`; mounted source artifact also available at `/Volumes/models/rdrr/gemma-3-270m-it-q4k-ehf16-af32` | failing | support matrix: node, 2026-03-11 | refreshed local artifact has `manifest.json`, bundled tokenizer files, 6 shards, `presetId: "gemma3"`, `schema: "doppler.execution/v0"`, `defaultKernelPath: "gemma3-q4k-dequant-f32a-online"`, and a fresh conversion report with `requiredInferenceFieldsArtifact.ok: true` and `executionV0GraphContractArtifact.ok: true` | `WS4` |
| `gemma-3-1b-it-q4k-ehf16-af32` | refreshed exact local artifact: `models/local/gemma-3-1b-it-q4k-ehf16-af32`; mounted source artifact also available at `/Volumes/models/rdrr/gemma-3-1b-it-q4k-ehf16-af32` | failing in support matrix, but contradicted by a coherent runtime report | Class `A`: [`reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json`](../reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json) on Apple M3 with coherent output and non-synthetic timings | refreshed local artifact has `manifest.json`, bundled tokenizer files, 16 shards, `presetId: "gemma3"`, `schema: "doppler.execution/v0"`, `defaultKernelPath: "gemma3-q4k-dequant-f32a-online"`, and a fresh conversion report with `requiredInferenceFieldsArtifact.ok: true` and `executionV0GraphContractArtifact.ok: true` | `WS4` |
| `translategemma-4b-it-q4k-ehf16-af32` | `/Volumes/models/rdrr/translategemma-4b-it-q4k-ehf16-af32`; no `models/local` artifact present | verified | support matrix: browser, 2026-03-06 | mounted artifact present with `manifest.json`, `origin.json`, `tokenizer.json`, `tokenizer.model`, and 47 shard files; manifest currently records `sessionDefaults.kvcache.layout: "paged"`. Latest sampled local report with that config is Class `C`: [`reports/translategemma-4b-it-q4k-ehf16-af32/2026-03-12T15-34-44.706Z.json`](../reports/translategemma-4b-it-q4k-ehf16-af32/2026-03-12T15-34-44.706Z.json) | `WS2` |
| `qwen-3-5-0-8b-q4k-ehaf16` | refreshed exact local artifact: `models/local/qwen-3-5-0-8b-q4k-ehaf16`; mounted comparison artifacts also exist at `/Volumes/models/rdrr/qwen-3-5-0-8b-wq4k-ef16-hf16-f16` and `/Volumes/models/rdrr/qwen-3-5-0-8b-wf16-ef16-hf16-f16` | failing | Class `A`: [`reports/qwen-3-5-0-8b-q4k-ehaf16/2026-03-12T16-45-00.422Z.json`](../reports/qwen-3-5-0-8b-q4k-ehaf16/2026-03-12T16-45-00.422Z.json) on Apple M3 with real load, real kernels, and incoherent newline/period output | refreshed exact local artifact has `manifest.json`, bundled tokenizer, 12 shards, `presetId: "qwen3"`, `schema: null`, `defaultKernelPath: null`, custom linear/full-attention `layerPattern`, and a fresh conversion report with `requiredInferenceFieldsArtifact.ok: true`, `executionContractArtifact.ok: true`, and `steps.total: 0` | `WS3.8` |
| `qwen-3-5-2b-q4k-ehaf16` | refreshed exact local artifact: `models/local/qwen-3-5-2b-q4k-ehaf16`; mounted comparison artifact also exists at `/Volumes/models/rdrr/qwen-3-5-2b-wq4k-ef16-hf16-f16` | failing | support matrix: browser, 2026-03-06 | refreshed exact local artifact has `manifest.json`, bundled tokenizer, 27 shards, `presetId: "qwen3"`, `schema: null`, `defaultKernelPath: null`, custom linear/full-attention `layerPattern`, and a fresh conversion report with `requiredInferenceFieldsArtifact.ok: true`, `executionContractArtifact.ok: true`, and `steps.total: 0` | `WS3` |
| `gemma-3-1b-it-f16-af32` | `models/local/gemma-3-1b-it-f16-af32` | verified | support matrix: browser, 2026-03-10 | sampled stale-manifest evidence | `WS5` |

Steps:

- [x] `WS0.1` Inventory the exact local artifact directories, manifests, tokenizer files, and shard sets for each target model.
- [x] `WS0.2` Classify all relevant report artifacts as Class `A`, `B`, or `C`. For [`reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json`](../reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json), independently verify the real-device and non-synthetic-timing criteria before accepting the Ground Truth item 6 pre-classification.
- [x] `WS0.3` Mark the known 2026-03-12 synthetic harness batch, plus any other reports with the same synthetic signature, as contract-only evidence unless they contain real device/provider info and non-synthetic timings.
- [x] `WS0.4` Record the latest real runtime claim for each model from the support matrix or a newer direct runtime smoke if one exists.
- [x] `WS0.5` Fill the inventory table and set one current next action per model.
- [x] `WS0.6` Update `Current resume point` to `WS1.1` when this workstream is complete.

Do not do:

1. Do not use canned outputs like `"The sky is blue."`, `"WebGPU"`, or `"Bonjour le monde."` as proof that a model actually ran correctly.
2. Do not mix metadata failures, manifest failures, and runtime-quality failures in the same conclusion.

## WS1: Artifact Freshness And Manifest Contract Repair

Status: `done`

Goal: re-establish whether the target local artifacts are current enough to support runtime investigation.

Entry gate:

- `WS0` inventory completed

Exit gate:

- each target artifact has one of:
  - a verified current manifest that satisfies the expected contract for that family, or
  - a fresh re-conversion using the checked-in conversion config, followed by a manifest re-check
  - a documented `non_refreshable` state when source checkpoints are unavailable, with downstream conclusions explicitly limited to the existing manifest state
- stale/incomplete artifacts are no longer being used as runtime evidence

Key files:

- [`src/converter/manifest-inference.js`](../src/converter/manifest-inference.js)
- [`tests/config/manifest-schema-contract.test.js`](../tests/config/manifest-schema-contract.test.js)
- [`tests/config/models-manifest-contract.test.js`](../tests/config/models-manifest-contract.test.js)
- [`tools/configs/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json`](../tools/configs/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json)
- [`tools/configs/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json`](../tools/configs/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json)
- [`tools/configs/conversion/gemma3/gemma-3-1b-it-f16-af32.json`](../tools/configs/conversion/gemma3/gemma-3-1b-it-f16-af32.json)
- [`tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json`](../tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json)
- [`tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16-af32.json`](../tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16-af32.json)
- [`tools/configs/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json`](../tools/configs/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json)

Required checks for every target artifact:

1. `manifest.json` exists and matches the expected model ID.
2. `inference.presetId` is present and correct.
3. `layerPattern` is complete and explicit.
4. nullable-but-required fields are explicit `null` where allowed rather than missing.
5. tokenizer config exists and points to a real tokenizer payload.
6. shard inventory matches the manifest.
7. `defaultKernelPath` and `schema` are recorded truthfully for that family, even when the correct value is `null`.

Family-specific guidance:

1. Do not require `execution-v0` or `steps.total > 0` as a universal gate. That is too strong.
2. For Qwen, record whether the real path is execution-v0 or legacy after reconversion; do not assume non-execution-v0 is automatically invalid.
3. For Gemma and TranslateGemma, check both manifest completeness and execution-v0 integrity where expected.

Steps:

- [x] `WS1.1` Inspect the current local manifests for all target models and record missing fields, explicit nulls, preset IDs, layer patterns, tokenizer paths, `schema`, and `defaultKernelPath`.
- [x] `WS1.2` Mark any artifact with missing required inference fields, missing tokenizer references, or shard/manifest mismatch as stale or invalid for runtime investigation.
- [x] `WS1.3` If source checkpoints are locally available, re-convert stale or incomplete artifacts using the checked-in conversion configs before making deeper runtime claims. If a source checkpoint is unavailable for a target model, mark that artifact `non_refreshable` and limit downstream conclusions to the existing manifest state rather than pretending the reconversion gate was satisfied.
- [x] `WS1.4` Re-run manifest contract checks after reconversion and capture the exact remaining gaps, if any.
- [x] `WS1.5` For Qwen, explicitly document whether the refreshed manifests still resolve to `defaultKernelPath: null`, whether `schema` remains null, and whether that is intentional for the actual runtime path.
- [x] `WS1.6` For Gemma and TranslateGemma, confirm execution-v0 presence and that required fields are explicit and complete.
- [x] `WS1.7` Update the inventory table in `WS0` with refreshed artifact state and set the next action for each model.
- [x] `WS1.8` Set `Current resume point` to `WS3.1`. Record in the progress log that `WS4.1` is also unblocked and may run concurrently, and that `WS2` remains independently available in parallel.

Do not do:

1. Do not start runtime-kernel theories while the local artifact is still manifest-incomplete.
2. Do not "fix" Qwen by forcing it into execution-v0 without first proving that the refreshed artifact still lacks a valid runtime path.
3. Do not close this workstream from synthetic report JSON alone; inspect the actual manifests directly.

## WS2: TranslateGemma Hardening And Metadata

Status: `in_progress`

Goal: close the explicit paged-layout hazard, align config with runtime behavior, and fix metadata truthfulness without over-claiming a runtime failure.

Entry gate: none

Exit gate:

- explicit `paged` plus `forceContiguousKVCache` is fail-closed
- TranslateGemma conversion config and runtime behavior agree
- catalog/support-matrix metadata is truthful
- any new verification claim is backed by real deterministic runtime evidence and human approval

Key files:

- [`tools/configs/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json`](../tools/configs/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json)
- [`src/inference/pipelines/text/init.js`](../src/inference/pipelines/text/init.js)
- [`src/config/presets/models/translategemma.json`](../src/config/presets/models/translategemma.json)
- [`tests/integration/translategemma-harness-default-prompt.test.js`](../tests/integration/translategemma-harness-default-prompt.test.js)
- [`models/catalog.json`](../models/catalog.json)
- [`docs/model-support-matrix.md`](./model-support-matrix.md)

Why this workstream exists:

1. The checked-in config explicitly requests `sessionDefaults.kvcache.layout: "paged"`.
2. The runtime currently prevents threshold auto-upgrade to paged when `forceContiguousKVCache` is true, but it does not reject an explicit paged request.
3. The debug log in [`init.js`](../src/inference/pipelines/text/init.js) currently emits `Layer pattern includes full-attention layers; forcing contiguous KV cache.` even though the current explicit-layout path does not rewrite `paged` back to `contiguous`.
4. TranslateGemma is still listed as verified, so this must be handled carefully as hardening plus truthfulness, not as a pre-declared runtime regression.
5. The checked-in paged conversion policy postdates the recorded 2026-03-06 verification, so `WS2.2` should optimize for the current checked-in policy rather than preserving an unproven paged verified path. If later real runtime evidence shows paged mixed-attention support is required, add an explicit config-level escape hatch instead of silently preserving current behavior.

Steps:

- [ ] `WS2.1` Add a targeted regression test that codifies the current baseline: explicit `layout: "paged"` is not rejected when `forceContiguousKVCache` is true. This test temporarily locks the pre-fix behavior and must be inverted or replaced with a fail-fast assertion in `WS2.2`.
- [ ] `WS2.2` Change the runtime path in [`init.js`](../src/inference/pipelines/text/init.js) so an explicit `paged` request for a full-attention or mixed-attention model fails fast with an actionable error instead of silently proceeding. Design the guard so a future explicit escape hatch can be added if later real runtime evidence requires it. Note: [`tests/integration/translategemma-harness-default-prompt.test.js`](../tests/integration/translategemma-harness-default-prompt.test.js) uses a harness override with a mock pipeline and supplied manifest, so it is a contract test and does not exercise `createKVCache`; update the assertion only if the contract policy changes. Also invert or remove the `WS2.1` pre-fix baseline test in this step.
- [ ] `WS2.3` Update the specific debug message in [`init.js`](../src/inference/pipelines/text/init.js) that currently says `Layer pattern includes full-attention layers; forcing contiguous KV cache.` so it reflects actual runtime behavior rather than implying a rewrite that did not happen.
- [ ] `WS2.4` After `WS2.2` lands, change the TranslateGemma conversion config from `layout: "paged"` to `layout: "contiguous"` so the manifest matches the actual supported policy.
- [ ] `WS2.4b` Re-convert a local TranslateGemma artifact using the updated conversion config from `WS2.4` and confirm the resulting manifest records `layout: "contiguous"`. If source weights are unavailable, mark `WS2.4b` and `WS2.7` through `WS2.9` blocked on source acquisition rather than proceeding with an older paged artifact; `WS2.5` and `WS2.6` may still proceed independently.
- [ ] `WS2.5` Inspect the TranslateGemma catalog entry and make `availability.local` and `baseUrl` truthful. If there is no stable local artifact path, mark it accordingly instead of claiming local availability.
- [ ] `WS2.6` If catalog metadata changed, run the support-matrix sync and verify the repo-visible mirror is consistent.
- [ ] `WS2.7` Ensure the runtime smoke uses the `WS2.4b` reconverted local artifact rather than an older paged artifact or an HF-hosted artifact built from older policy.
- [ ] `WS2.8` Run a real deterministic TranslateGemma smoke using the local artifact and capture prompt, output, surface, and config. Do not rely on synthetic harness output.
- [ ] `WS2.9` Before updating any verification claim, stop and ask for human review of the real observed output.

Do not do:

1. Do not claim TranslateGemma is broken solely because the paged-layout hazard exists.
2. Do not claim the hazard is harmless solely because a prior browser verification exists.
3. Do not update support claims without a human-reviewed runtime smoke.

## WS3: Qwen Runtime-Path And Linear-Attention Investigation

Status: `validated`

Goal: determine whether Qwen fails because of stale artifacts, missing or ambiguous runtime-path anchoring, linear-attention correctness, or some combination.

Entry gate:

- `WS1` complete for both Qwen model IDs

Exit gate:

- refreshed Qwen artifacts are inspected and no longer stale [Class `B` sufficient]
- the actual dispatch path is documented truthfully [Class `B` sufficient]
- linear-attention and Qwen-specific semantic risks are narrowed to a short list of evidence-backed candidates [Class `B` sufficient for narrowing; Class `A` required for root-cause confirmation via the smoke gate below]
- at least one real deterministic smoke exists after the artifact refresh [Class `A` required]

Key files:

- [`src/config/presets/models/qwen3.json`](../src/config/presets/models/qwen3.json)
- [`tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json`](../tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json)
- [`tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16-af32.json`](../tools/configs/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16-af32.json)
- [`tools/configs/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json`](../tools/configs/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json)
- [`tools/configs/conversion/qwen3/qwen-3-5-0-8b-f16.json`](../tools/configs/conversion/qwen3/qwen-3-5-0-8b-f16.json)
- `models/local/qwen-3-5-0-8b-f16/manifest.json`
- [`src/inference/pipelines/text/linear-attention.js`](../src/inference/pipelines/text/linear-attention.js)
- [`src/inference/pipelines/text/attention/run.js`](../src/inference/pipelines/text/attention/run.js)
- [`tests/inference/linear-attention-contract.test.js`](../tests/inference/linear-attention-contract.test.js)
- [`tests/inference/qwen-manifest-completeness.test.js`](../tests/inference/qwen-manifest-completeness.test.js)
- [`tests/inference/qwen-rope-runtime-config.test.js`](../tests/inference/qwen-rope-runtime-config.test.js)
- [`tests/inference/qwen-linear-norm-offset.test.js`](../tests/inference/qwen-linear-norm-offset.test.js)
- [`tests/integration/qwen-linear-attention-regression.test.js`](../tests/integration/qwen-linear-attention-regression.test.js)

Important guardrails:

1. Qwen may legitimately use a non-execution-v0 path. Do not assume execution-v0 is mandatory.
2. Standard attention and linear attention use confirmed different normalization mechanisms. Standard attention applies learned Q/K normalization via checkpoint weights, while linear attention applies in-kernel L2 normalization as part of the delta-net path. The remaining question is whether Qwen's linear layers require additional learned Q/K normalization on top of that in-kernel normalization.
3. `partialRotaryFactor` is not automatically a concern for linear layers. Verify where RoPE is actually applied before adding work there.
4. Any missing-kernel-path theory must be tied to the real post-refresh dispatch path, not only to old conversion configs.

Steps:

- [x] `WS3.1` After `WS1`, inspect the refreshed manifests for both Qwen Q4K models and the local F16 comparison artifact. Record `presetId`, `schema`, `defaultKernelPath`, `layerPattern`, tokenizer state, and any explicit linear-attention fields so Q4K-specific versus family-wide issues can be separated early.
- [x] `WS3.2` Determine the actual runtime dispatch path for the refreshed artifacts: execution-v0, kernel-path-derived execution, or legacy path. Document it explicitly.
- [x] `WS3.3` If `WS3.2` shows that the actual dispatch path cannot reach the intended linear-attention kernels or selects the wrong runtime path, make the smallest manifest-first fix possible and add tests that prove the path is explicit rather than implicit. If the legacy path dispatches correctly with `defaultKernelPath: null`, record that finding and move to `WS3.4`.
- [x] `WS3.4` Re-check the Qwen-specific config semantics already present in [`qwen3.json`](../src/config/presets/models/qwen3.json): `queryKeyNorm`, `mropeInterleaved`, `mropeSection`, `partialRotaryFactor`, and `rmsNormWeightOffset`.
- [x] `WS3.5` Verify whether Qwen linear layers are supposed to use learned Q/K normalization weights, pure in-kernel L2 normalization, or both. Do not assume equivalence with standard-attention RMSNorm.
- [x] `WS3.6` Audit `linearNormMode` resolution for the refreshed artifacts. If inference from weight shape is ambiguous, move that field to explicit manifest output instead of relying on shape heuristics.
- [x] `WS3.7` If a local reference environment and source weights are available, run a real deterministic reference comparison against a known-good implementation and isolate divergence to one of the following. If that environment is unavailable, record the skip and proceed to `WS3.8`:
  - manifest/runtime-path setup
  - linear-attention state initialization
  - linear-attention math
  - Qwen-specific normalization semantics
  - tokenizer or prompt-format mismatch
- [x] `WS3.8` If `WS3.7` identified a specific failure class, strengthen tests so that failure becomes reproducible without relying on a vague "incoherent output" description. If `WS3.7` was skipped, add a focused test stub or TODO anchored to the strongest candidate from `WS3.4` through `WS3.6`.
- [ ] `WS3.9` Re-run a real deterministic Qwen smoke after the fix and record prompt, config, output, and surface.
- [ ] `WS3.10` Stop for human review before updating any catalog or support-matrix claim.

Do not do:

1. Do not force a Qwen-specific kernel path or execution-v0 graph merely because the current conversion config shows `defaultKernelPath: null`.
2. Do not close Qwen from old synthetic report outputs.
3. Do not change linear-attention math until artifact freshness and runtime-path clarity are established first.

## WS4: Gemma Q4K Conversion And Numeric Triage

Status: `in_progress`

Goal: separate artifact/conversion integrity from runtime-path divergence for the failing Gemma 3 Q4K models.

Entry gate:

- `WS1` complete for the Gemma Q4K targets

Exit gate:

- the Gemma Q4K failures are narrowed to either conversion/artifact integrity or runtime math/path
- the strongest remaining hypotheses are backed by direct evidence rather than contrast-only reasoning
- at least one real deterministic smoke exists after any artifact refresh

Key files:

- [`docs/model-support-matrix.md`](./model-support-matrix.md)
- [`tools/configs/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json`](../tools/configs/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json)
- [`tools/configs/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json`](../tools/configs/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json)
- [`src/config/presets/kernel-paths/gemma3-q4k-dequant-f32a-online.json`](../src/config/presets/kernel-paths/gemma3-q4k-dequant-f32a-online.json)
- [`src/config/presets/kernel-paths/gemma3-q4k-dequant-f16a-online.json`](../src/config/presets/kernel-paths/gemma3-q4k-dequant-f16a-online.json)
- relevant Q4K kernels under `src/gpu/kernels/**`

Required reasoning standard:

1. Use the conversion triage protocol in `doppler/AGENTS.md`.
2. Do not label the problem as "runtime" until source dtypes, manifest fields, shard integrity, and sampled numeric sanity all pass.
3. Do not label the problem as "conversion" until one of those checks fails.
4. The F32a vs F16a kernel-math hypothesis (`H-GEMMA-F32A`) is disproved. If this comparison is still useful, scope it to the orchestration layer under `H-GEMMA-F32A-ORCH`.

Steps:

- [x] `WS4.1` Inspect the refreshed Gemma Q4K manifests and confirm `presetId`, quantization settings, `quantizationInfo` including `layout`, `layerPattern`, `defaultKernelPath`, and execution-v0 state. Note: per Ground Truth item 6, the 1B Q4K model has Class `A` evidence of coherent output on at least one prompt, so the Q4K path is not categorically broken across both Gemma targets.
- [x] `WS4.2` Verify source checkpoint dtypes for the failing models and record the expected dtype mix. If source checkpoints are not available locally, inspect the corresponding Hugging Face metadata or checkpoint headers where possible and record any remaining access gap explicitly rather than leaving this step implicitly skipped.
- [x] `WS4.3` Verify shard integrity by sampling hashes against the manifest.
- [x] `WS4.4` Sample numeric sanity by comparing selected source tensor values with converted bytes and dequantized values.
- [x] `WS4.5` If any of `WS4.2` through `WS4.4` fails, fix the conversion or artifact issue first and re-run the checks before touching runtime math.
- [ ] `WS4.6` If conversion integrity passes, audit the runtime Q4K path:
  - dequant math
  - `quantizationInfo.layout`
  - kernel-path selection
  - subgroup vs no-subgroups behavior (note: `gemma3-q4k-dequant-f32a-nosubgroups` is also the current LFM2 fallback path; escalate any finding here to `WS5.C`)
  - first-layer or early-layer numeric divergence
- [ ] `WS4.7` If conversion and artifact integrity pass, compare F32a and F16a behavior at the orchestration layer only: activation-dtype propagation, buffer allocation sizing, weight-buffer tagging, and any runtime compile differences. Do not treat the paired WGSL kernels themselves as the primary comparison axis.
- [ ] `WS4.8` Add a focused test or debug harness that would catch the identified Q4K failure class again.
- [ ] `WS4.9` Re-run a real deterministic Gemma Q4K smoke and capture prompt, output, and surface.
- [ ] `WS4.10` Stop for human review before changing support claims.

Do not do:

1. Do not close this workstream from support-matrix notes alone.
2. Do not promote "small models are too lossy under Q4K" from a plausible explanation to a fact without numeric evidence.
3. Do not treat a clean execution-v0 graph as proof that the underlying math is correct.
4. Do not keep treating execution-v0 vs explicit `kernelPath` selection as the primary 270M branch after the 2026-03-12 probe runs. That branch is now materially weaker than non-selection orchestration or model-specific numeric drift.

## WS5: Maintenance Appendix

Status: `ready`

Goal: isolate side issues so they are handled rigorously without derailing the primary workstreams.

Entry gate: none

Exit gate:

- each appendix item is either fixed, parked with evidence, or explicitly deprioritized

### WS5.A: Gemma 3 1B F16 Sampled Stale Artifact

Status: `ready`

Why it exists:

Sampled contract-only report evidence suggests at least one local `gemma-3-1b-it-f16-af32` artifact may predate the current manifest completeness contract, even though the support matrix shows a real browser verification on 2026-03-10.

Steps:

- [ ] `WS5.A.1` Inspect the actual local `gemma-3-1b-it-f16-af32` manifest rather than relying only on synthetic report artifacts.
- [ ] `WS5.A.2` If the local artifact is stale, re-convert it and re-run manifest checks.
- [ ] `WS5.A.3` Do not let this side issue override the real verified status until real runtime evidence says otherwise.

### WS5.B: Gemma2 Execution-v0 Slot-Graph Regression

Status: `ready`

Why it exists:

A Class `C` synthetic report suggests a possible `gemma2` execution-v0 graph bug where a step reads `attn_q` before it is produced. Source: [`reports/gemma2-sharded-index/2026-03-12T13-48-55.531Z.json`](../reports/gemma2-sharded-index/2026-03-12T13-48-55.531Z.json). The report's runtime metrics are not admissible, but the slot-graph contract artifact inside it is still useful Class `B` evidence until reproduced or disproved. This is a converter regression track, not part of the main model-failure cluster.

Key files:

- [`tools/configs/conversion/gemma2/gemma2-template-f16.json`](../tools/configs/conversion/gemma2/gemma2-template-f16.json)
- [`reports/gemma2-sharded-index/2026-03-12T13-48-55.531Z.json`](../reports/gemma2-sharded-index/2026-03-12T13-48-55.531Z.json)

Steps:

- [ ] `WS5.B.0` Determine whether `gemma2-sharded-index` is an expected error-surface fixture or a real converter-regression target. Check the fixture or harness definition first. If `ok: false` is expected behavior, mark this sub-workstream `done` with a note that the synthetic fixture is behaving as designed, and skip `WS5.B.1` through `WS5.B.3`.
- [ ] `WS5.B.1` If `WS5.B.0` shows this is a real regression target, start from the checked-in Gemma2 conversion template, identify the concrete Gemma2 model/config pair that reproduces the `attn_q` ordering bug, and then reproduce the slot-graph failure with the current converter while capturing the exact failing step order. If reproduction fails and `WS5.B.0` does not establish that failure was expected, mark the sub-workstream `done` with a note that the synthetic report was a likely false positive.
- [ ] `WS5.B.2` Fix the graph builder or step ordering in a manifest-first way.
- [ ] `WS5.B.3` Add a regression test that fails on the old slot ordering and passes on the corrected graph.

### WS5.C: LFM2 Runtime-Path Verification

Status: `ready`

Why it exists:

The current registry confirms `lfm2-q4k-dequant-f32a-online` exists but no `lfm2-q4k-dequant-f32a-nosubgroups` path exists, and the inference kernel-path rules already map LFM2 to `gemma3-q4k-dequant-f32a-nosubgroups` on non-subgroup devices and on finiteness fallback. Gemma3 kernels do not own LFM2's conv-layer path, so the remaining work is to verify runtime impact and replace the fallback with an LFM2-specific path.

Steps:

- [ ] `WS5.C.1` Record the existing rule and registry evidence that LFM2 falls back to `gemma3-q4k-dequant-f32a-nosubgroups` and that no LFM2 no-subgroups path exists.
- [ ] `WS5.C.2` Verify the runtime impact of that fallback on LFM2's conv layers and shared full-attention layers.
- [ ] `WS5.C.3` If the fallback is wrong, add the correct LFM2-specific path and rule wiring.
- [ ] `WS5.C.4` Verify the conv-layer dispatch path for LFM2: confirm `doConv()` receives the correct `convInProj`, `convOutProj`, and `convKernel` weights for all conv layers, and confirm the `full_attention` layers stay on the standard attention path.
- [ ] `WS5.C.5` Only after the path is verified, run a deterministic real smoke and decide whether LFM2 belongs in the catalog.

### WS5.D: Catalog And Support-Matrix Hygiene

Status: `ready`

Why it exists:

The repo-visible mirror must stay truthful even when a runtime issue is still under investigation.

Steps:

- [ ] `WS5.D.1` Audit `models/catalog.json` for false `availability.local` or missing `baseUrl` values on tracked models, excluding TranslateGemma if `WS2.5` is already complete.
- [ ] `WS5.D.2` Reconcile any release-claim or support-matrix drift with the canonical external registry.
- [ ] `WS5.D.3` Run the support-matrix sync only after the catalog mirror is truthful.
- [ ] `WS5.D.4` Do not upgrade a model to verified without a human-reviewed real smoke result.

## Validation Requirements Before Calling Anything Fixed

The minimum acceptable closeout for a model-correctness issue is:

1. Fresh or directly verified artifact state.
2. Real deterministic runtime smoke with:
   - actual model load
   - actual device/provider info
   - actual prompt and output
   - recorded config and surface
3. Any contract or config changes covered by tests.
4. Human review of the observed output before updating `models/catalog.json` or `docs/model-support-matrix.md`.
5. Support-matrix sync run if catalog metadata changed.

## Progress Log Protocol

Append one row every time you finish a meaningful step, disprove a hypothesis, or hit a blocker.

Required fields:

1. UTC date/time
2. actor
3. workstream and step ID
4. status change
5. evidence class used
6. short note
7. next resume point

Actor naming convention:

- use a stable format such as `agent/<name-or-id>`

Template:

| UTC | Actor | Step | Status change | Evidence | Note | Next resume point |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-03-12T00:00:00Z | example | `WS0.1` | `ready -> done` | `B` | Inventoried local artifacts for five target models | `WS0.2` |

## Progress Log

| UTC | Actor | Step | Status change | Evidence | Note | Next resume point |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-03-12T00:00:00Z | agent/codex | plan creation | n/a | `B` | Created the canonical model-failure action plan with workstreams, gates, and resume rules | `WS0.1` |
| 2026-03-12T14:39:04Z | agent/codex | plan review integration | n/a | `B` | Integrated accepted review findings from batches 2 through 4, including LFM2 fallback evidence, deterministic resume rules, Qwen key-file expansion, and TranslateGemma artifact sequencing | `WS0.1` |
| 2026-03-12T15:05:42Z | agent/codex | plan review integration | n/a | `B` | Merged validated follow-up suggestions on WS2 baseline-test disposal, Gemma manifest wording, and Gemma2 fixture gating while rejecting duplicate or repo-contradicted edits | `WS0.1` |
| 2026-03-12T15:06:59Z | agent/codex | plan review integration | n/a | `B` | Normalized WS5.B closeout semantics to use only legend-defined statuses and preserved the fixture-vs-regression gate | `WS0.1` |
| 2026-03-12T16:12:12Z | agent/codex | `WS0.1` | `ready -> done` | `B` | Inventoried mounted and local target artifacts; Gemma Q4K and TranslateGemma now have mounted artifacts, while Qwen Q4K remains absent locally | `WS0.2` |
| 2026-03-12T16:12:12Z | agent/codex | `WS0.2`-`WS0.6` | `ready -> done` | `A`,`B`,`C` | Classified the real Gemma 1B runtime report, marked null-device fixed-timing reports as synthetic Class `C`, aligned latest runtime claims with the support matrix, and completed the inventory table | `WS1.1` |
| 2026-03-12T16:18:42Z | agent/codex | `WS1.1`-`WS1.2` | `ready -> done` | `B` | Inspected the mounted Gemma and TranslateGemma manifests plus Qwen comparison artifacts; exact Qwen target IDs remain absent, but source checkpoints for all primary families are available on the mounted volume | `WS1.3` |
| 2026-03-12T16:39:24Z | agent/codex | `WS1.3`-`WS1.7` | `ready -> done` | `B` | Refreshed exact local artifacts for both Qwen Q4K targets and both Gemma Q4K targets, re-checked manifests, and recorded the remaining family-specific gaps: Qwen stays `schema: null`/`defaultKernelPath: null`, while Gemma remains execution-v0 complete | `WS1.8` |
| 2026-03-12T16:39:24Z | agent/codex | `WS1.8` | `ready -> done` | `B` | Closed the artifact-freshness gate; `WS3.1` is now the primary resume point, `WS4.1` is also unblocked, and `WS2` remains independently available | `WS3.1` |
| 2026-03-12T16:44:57Z | agent/codex | `WS3.1`-`WS3.3` | `ready -> done` | `B` | Compared refreshed Qwen Q4K manifests with the local F16 baseline and traced node debug/test-model dispatch: no execution-v0, no model-level kernel path, legacy layer-pattern routing still reaches `runLinearAttentionLayer()`, and generic kernel auto-selection happens below that | `WS3.4` |
| 2026-03-12T16:46:42Z | agent/codex | `WS3.4`-`WS3.6` | `ready -> done` | `B` | Re-checked Qwen preset semantics against refreshed manifests, confirmed `linearNormMode: "shared"` is explicit in both refreshed Qwen Q4K artifacts, and verified from source checkpoint indexes that learned `q_norm`/`k_norm` weights appear on full-attention `self_attn` layers only, not on linear-attention layers | `WS3.7` |
| 2026-03-12T16:54:51Z | agent/codex | `WS3` runtime smoke | `n/a` | `A` | Real node/WebGPU debug smoke for refreshed `qwen-3-5-0-8b-q4k-ehaf16` loaded successfully on Apple M3 and produced degenerate newline/period output, confirming the failure survives artifact refresh | `WS3.7` |
| 2026-03-12T16:54:51Z | agent/codex | `WS3` F16 control smoke | `n/a` | `A` | Real node/WebGPU debug smoke for local `qwen-3-5-0-8b-f16` produced multilingual gibberish on the same prompt, weakening Q4K-only explanations for Qwen | `WS3.7` |
| 2026-03-12T16:54:51Z | agent/codex | `WS3.7` | `ready -> done` | `B` | Attempted a local Hugging Face reference run against the Qwen 0.8B source checkpoint, but installed `transformers` does not recognize `model_type=\"qwen3_5\"` and the local snapshot ships no remote-code module; recorded the skip and moved to test-strengthening work | `WS3.8` |
| 2026-03-12T17:02:15Z | agent/codex | `WS3.8` | `ready -> done` | `B` | Strengthened Qwen contract coverage by asserting refreshed F16 and exact Q4K manifests share the expected hybrid layer topology, that full-attention Q/K norm weights stay confined to `self_attn` layers, and that linear-attention norm weights resolve to `shared`; targeted unit tests passed | `WS4.1` |
| 2026-03-12T17:10:23Z | agent/codex | `WS4.1`-`WS4.5` | `ready -> done` | `A`,`B` | Confirmed refreshed Gemma 270M and 1B Q4K manifests are symmetric execution-v0 artifacts, verified BF16 source checkpoints, matched sampled local shard hashes, and reproduced exact `q_proj` Q4K block bytes from source BF16 weights, clearing conversion/artifact integrity as the primary blocker | `WS4.6` |
| 2026-03-12T17:10:23Z | agent/codex | `WS4` runtime smokes | `n/a` | `A` | Fresh Apple M3 node/WebGPU smokes split by model on the same execution-v0 path: `gemma-3-1b-it-q4k-ehf16-af32` stayed coherent while `gemma-3-270m-it-q4k-ehf16-af32` remained incoherent counting text, narrowing the issue toward model-specific runtime or numeric sensitivity rather than generic Gemma Q4K corruption | `WS4.6` |
| 2026-03-12T17:13:18Z | agent/codex | `WS4.6` | `ready -> in_progress` | `A`,`B` | Forced the 270M model onto `gemma3-q4k-dequant-f32a-nosubgroups`: the override failed closed until runtime `f32` activation/output dtypes were pinned, and the corrected run still produced incoherent output, weakening a subgroup-only explanation while keeping orchestration-layer dtype handling in scope | `WS4.6` |
| 2026-03-12T17:31:26Z | agent/codex | `WS4.6` | `in_progress -> in_progress` | `A` | Ran fresh local-file node/WebGPU probe smokes on `gemma-3-270m-it-q4k-ehf16-af32` for both the manifest-inline execution-v0 path and an explicit `gemma3-q4k-dequant-f32a-nosubgroups` override. Layer-0 probe values and sampled first-token logits matched to logged precision on one-token runs, and a three-token deterministic decode produced the same `"\nDeterminate"` prefix with matching layer-0 decode probes, so kernel-path selection under execution-v0 vs explicit override is no longer the lead WS4.6 branch | `WS4.6` |
| 2026-03-12T18:03:00Z | agent/codex | `WS4.6` | `in_progress -> in_progress` | `B` | Fixed a silent raw-buffer dtype propagation gap: `getWeightDtype()` now returns tagged dtype metadata for raw `GPUBuffer` weights, text embedding entrypoints use that accessor instead of wrapper-only metadata, and a new regression test proves QKV fusion keeps tagged `f16` semantics even when pooled raw buffer sizes would otherwise bias size-based inference toward `f32`. Targeted tests passed, but no post-fix Class `A` Gemma 270M runtime smoke has been run yet | `WS4.6` |
| 2026-03-12T18:10:00Z | agent/codex | `WS4.6` | `in_progress -> in_progress` | `A`,`B` | Replayed the 270M local-file prefill probe on the patched build. The refreshed path still loads embeddings as `WeightBuffer`, gathers with `embeddingDtype=f16`, emits the same one-token output (`"\n"`), and matches the pre-fix `embed_out`, layer-0 `q_proj`, `attn_out`, `post_attn`, `ffn_in`, `ffn_out`, `layer_out`, `final_norm`, and sampled logits to logged precision. The raw-buffer dtype fix is therefore real but not the active 270M root cause on refreshed artifacts | `WS4.6` |
| 2026-03-12T18:18:00Z | agent/codex | `WS4.6` | `in_progress -> in_progress` | `B` | Fixed the BF16 matmul-weight policy gap so `runtime.inference.compute.keepF32Weights=true` now keeps BF16-origin matmul weights in `f32` instead of silently converting them to `f16` on `shader-f16` devices. Added loader regression coverage for both branches and re-ran the targeted loader/inference contract tests. This closes a real debug/compat bug, but it has not yet been tied to a refreshed Class `A` model outcome | `WS4.6` |
| 2026-03-12T19:13:40Z | agent/codex | `WS4.6` | `in_progress -> in_progress` | `A`,`B` | Reconstructed layer-0 `attn_normed` offline from the real prompt tokenization, local embeddings, and `input_layernorm.weight`, then compared sampled `q_proj` outputs against CPU matmul from both source BF16 weights and stored exact-Q4K weights. The source path matched runtime to ~`4e-5` max error and the exact-Q4K path matched to ~`7e-3`, so layer-0 `q_proj` divergence is now explained by legitimate quantization error rather than a first-projection runtime bug | `WS4.6` |
| 2026-03-12T19:14:37Z | agent/codex | `WS4.6` | `in_progress -> in_progress` | `B` | Replayed the same raw prompt offline through both local F16 and exact-Q4K artifacts for Gemma `270M` and `1B`. The legitimate layer-0 `q_proj` delta induced by Q4K is much larger on `270M` than on `1B` (mean absolute delta about `1.40` vs `0.31`, sampled-dim drift up to about `6.07` vs `0.27`), which strengthens the remaining “model-specific numeric sensitivity / later-stage amplification” branch | `WS4.6` |
