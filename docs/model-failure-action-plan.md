# Model Failure Action Plan

Last updated: 2026-03-12T15:06:59Z
Plan status: active
Current resume point: `WS0.1`
Current highest-priority ready step: `WS0.1`

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
4. The sampled local report artifacts from 2026-03-12 are mostly Class `C` evidence. They are useful for contract artifacts, but not for generation quality or performance claims.
5. The earlier "Gemma F32a path is the prime suspect because TranslateGemma uses F16a" story is no longer a lead explanation. Kernel-level review reported no arithmetic divergence between the paired F32a and F16a WGSL paths beyond I/O and storage dtype differences, so any remaining F32a vs F16a comparison belongs at the orchestration layer rather than kernel math.
6. There is real Class `A` evidence that `gemma-3-1b-it-q4k-ehf16-af32` produced coherent output on at least one prompt on 2026-03-11 in [`reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json`](../reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json). This contradicts the support-matrix "failing" summary and means the 1B Q4K path must not be treated as categorically broken.
7. Qwen remains the strongest repo-backed case for a family-specific runtime correctness issue centered on linear attention and Qwen-specific semantics.
8. Sampled local artifacts for at least some Qwen and Gemma models appear stale or incomplete against the current manifest-first contract and must be refreshed or re-verified before deeper runtime conclusions.

## Hypothesis Register

| ID | Hypothesis | Status | Current read | Next action |
| --- | --- | --- | --- | --- |
| `H-GEMMA-Q4K` | Gemma 3 Q4K failures are in the Q4K artifact/runtime path, not generic Gemma 3 execution | `in_progress` | Strong repo support; exact sub-cause still unknown | Run conversion triage and then runtime-path audit |
| `H-GEMMA-F32A` | Gemma F32a vs F16a kernel divergence is the primary root cause | `disproved` | Kernel-level review reported paired WGSL paths with matching arithmetic structure and f32 accumulation; only storage and I/O dtype differ | N/A |
| `H-GEMMA-F32A-ORCH` | Gemma F32a vs F16a orchestration-layer dtype propagation or buffer handling difference causes divergence | `needs_recheck` | Still plausible at the orchestration layer even if kernel math is not the cause | Check activation-dtype flow, buffer sizing, and weight-buffer tagging |
| `H-TG-PAGED` | Explicit `paged` KV layout bypasses the contiguous-only intent for Gemma 3 mixed-attention models | `validated` | Confirmed by current `init.js` control flow; `WS2.1` should codify this as the pre-fix baseline | Add fail-fast and then align config |
| `H-TG-BROKEN` | TranslateGemma is fundamentally broken in Doppler | `needs_recheck` | Not supported by current repo evidence | Treat as unproven until real runtime repro exists |
| `H-QWEN-LA` | Qwen 3.5 failure is driven by linear-attention correctness or Qwen-specific runtime semantics | `in_progress` | Best repo-backed runtime hypothesis | Refresh artifacts, inspect dispatch path, then do numeric/reference work |
| `H-QWEN-STALE` | Sampled local Qwen artifacts are stale or manifest-incomplete relative to the current contract | `validated` | Strong contract-only evidence | Re-convert or directly re-verify manifests before runtime conclusions |
| `H-TG-META` | TranslateGemma local-loading/catalog metadata is inconsistent | `validated` | Known metadata issue | Fix catalog truthfully and re-sync support matrix if needed |
| `H-LFM2-FALLBACK` | LFM2 no-subgroups path silently falls back to the wrong kernel family | `validated` | Confirmed by `kernel-path.rules.json`: LFM2 falls back to `gemma3-q4k-dequant-f32a-nosubgroups` on non-subgroup devices and on finiteness fallback, while the registry lacks an LFM2 no-subgroups path | Create an LFM2-specific no-subgroups kernel path and verify conv-layer dispatch |

## Master Status Board

| Workstream | Status | Priority | Depends on | Exit gate |
| --- | --- | --- | --- | --- |
| `WS0` Evidence normalization and inventory | `ready` | P0 | none | Target artifacts, evidence classes, and current failure claims are pinned |
| `WS1` Artifact freshness and manifest contract repair | `blocked` | P1 | `WS0` | Target artifacts are current enough for runtime investigation |
| `WS2` TranslateGemma hardening and metadata | `ready` | P1 | none | Explicit paged-layout hazard is fail-closed and config/catalog are aligned |
| `WS3` Qwen runtime-path and linear-attention investigation | `blocked` | P1 | `WS1` | Real runtime behavior is explained and validated with real evidence |
| `WS4` Gemma Q4K conversion and numeric triage | `blocked` | P1 | `WS1` | Conversion/runtime split is resolved with real evidence |
| `WS5` Maintenance appendix: Gemma 1B F16, Gemma2, LFM2, catalog cleanup | `ready` | P2 | none | Side issues are either fixed, parked with evidence, or explicitly deprioritized |

## Order Of Operations

1. Finish `WS0` before drawing any new runtime conclusion.
2. Finish the artifact-freshness gate in `WS1` before closing anything in `WS3` or `WS4`.
3. `WS2` may run in parallel because the TranslateGemma paged-layout guard issue is a concrete code-path problem regardless of whether a fresh runtime repro exists today.
4. `WS5` must not block `WS2`, `WS3`, or `WS4`. Treat it as a maintenance appendix unless it produces new evidence that changes the primary workstreams.
5. Do not update release-facing metadata until a human has reviewed coherent output from a real deterministic smoke.

## WS0: Evidence Normalization And Inventory

Status: `ready`

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
| `gemma-3-270m-it-q4k-ehf16-af32` | fill in | failing | support matrix: node, 2026-03-11 | fill in | `WS1` then `WS4` |
| `gemma-3-1b-it-q4k-ehf16-af32` | fill in | failing in support matrix, but contradicted by a coherent runtime report | Class `A`: [`reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json`](../reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json) on Apple M3 with coherent output and non-synthetic timings | fill in | `WS1` then `WS4` |
| `translategemma-4b-it-q4k-ehf16-af32` | fill in | verified | support matrix: browser, 2026-03-06 | paged-layout contract evidence | `WS2` |
| `qwen-3-5-0-8b-q4k-ehaf16` | fill in | failing | support matrix: browser, 2026-03-06 | sampled stale-manifest evidence | `WS1` then `WS3` |
| `qwen-3-5-2b-q4k-ehaf16` | fill in | failing | support matrix: browser, 2026-03-06 | fill in | `WS1` then `WS3` |
| `gemma-3-1b-it-f16-af32` | fill in | verified | support matrix: browser, 2026-03-10 | sampled stale-manifest evidence | `WS5` |

Steps:

- [ ] `WS0.1` Inventory the exact local artifact directories, manifests, tokenizer files, and shard sets for each target model.
- [ ] `WS0.2` Classify all relevant report artifacts as Class `A`, `B`, or `C`. For [`reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json`](../reports/gemma-3-1b-it-q4k-ehf16-af32/2026-03-11T17-22-04.007Z.json), independently verify the real-device and non-synthetic-timing criteria before accepting the Ground Truth item 6 pre-classification.
- [ ] `WS0.3` Mark the known 2026-03-12 synthetic harness batch, plus any other reports with the same synthetic signature, as contract-only evidence unless they contain real device/provider info and non-synthetic timings.
- [ ] `WS0.4` Record the latest real runtime claim for each model from the support matrix or a newer direct runtime smoke if one exists.
- [ ] `WS0.5` Fill the inventory table and set one current next action per model.
- [ ] `WS0.6` Update `Current resume point` to `WS1.1` when this workstream is complete.

Do not do:

1. Do not use canned outputs like `"The sky is blue."`, `"WebGPU"`, or `"Bonjour le monde."` as proof that a model actually ran correctly.
2. Do not mix metadata failures, manifest failures, and runtime-quality failures in the same conclusion.

## WS1: Artifact Freshness And Manifest Contract Repair

Status: `blocked`
Blocked on: `WS0`

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

- [ ] `WS1.1` Inspect the current local manifests for all target models and record missing fields, explicit nulls, preset IDs, layer patterns, tokenizer paths, `schema`, and `defaultKernelPath`.
- [ ] `WS1.2` Mark any artifact with missing required inference fields, missing tokenizer references, or shard/manifest mismatch as stale or invalid for runtime investigation.
- [ ] `WS1.3` If source checkpoints are locally available, re-convert stale or incomplete artifacts using the checked-in conversion configs before making deeper runtime claims. If a source checkpoint is unavailable for a target model, mark that artifact `non_refreshable` and limit downstream conclusions to the existing manifest state rather than pretending the reconversion gate was satisfied.
- [ ] `WS1.4` Re-run manifest contract checks after reconversion and capture the exact remaining gaps, if any.
- [ ] `WS1.5` For Qwen, explicitly document whether the refreshed manifests still resolve to `defaultKernelPath: null`, whether `schema` remains null, and whether that is intentional for the actual runtime path.
- [ ] `WS1.6` For Gemma and TranslateGemma, confirm execution-v0 presence and that required fields are explicit and complete.
- [ ] `WS1.7` Update the inventory table in `WS0` with refreshed artifact state and set the next action for each model.
- [ ] `WS1.8` Set `Current resume point` to `WS3.1`. Record in the progress log that `WS4.1` is also unblocked and may run concurrently, and that `WS2` remains independently available in parallel.

Do not do:

1. Do not start runtime-kernel theories while the local artifact is still manifest-incomplete.
2. Do not "fix" Qwen by forcing it into execution-v0 without first proving that the refreshed artifact still lacks a valid runtime path.
3. Do not close this workstream from synthetic report JSON alone; inspect the actual manifests directly.

## WS2: TranslateGemma Hardening And Metadata

Status: `ready`

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

Status: `blocked`
Blocked on: `WS1`

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
- [`tests/inference/qwen-rope-runtime-config.test.js`](../tests/inference/qwen-rope-runtime-config.test.js)
- [`tests/inference/qwen-linear-norm-offset.test.js`](../tests/inference/qwen-linear-norm-offset.test.js)
- [`tests/integration/qwen-linear-attention-regression.test.js`](../tests/integration/qwen-linear-attention-regression.test.js)

Important guardrails:

1. Qwen may legitimately use a non-execution-v0 path. Do not assume execution-v0 is mandatory.
2. Standard attention and linear attention use confirmed different normalization mechanisms. Standard attention applies learned Q/K normalization via checkpoint weights, while linear attention applies in-kernel L2 normalization as part of the delta-net path. The remaining question is whether Qwen's linear layers require additional learned Q/K normalization on top of that in-kernel normalization.
3. `partialRotaryFactor` is not automatically a concern for linear layers. Verify where RoPE is actually applied before adding work there.
4. Any missing-kernel-path theory must be tied to the real post-refresh dispatch path, not only to old conversion configs.

Steps:

- [ ] `WS3.1` After `WS1`, inspect the refreshed manifests for both Qwen Q4K models and the local F16 comparison artifact. Record `presetId`, `schema`, `defaultKernelPath`, `layerPattern`, tokenizer state, and any explicit linear-attention fields so Q4K-specific versus family-wide issues can be separated early.
- [ ] `WS3.2` Determine the actual runtime dispatch path for the refreshed artifacts: execution-v0, kernel-path-derived execution, or legacy path. Document it explicitly.
- [ ] `WS3.3` If `WS3.2` shows that the actual dispatch path cannot reach the intended linear-attention kernels or selects the wrong runtime path, make the smallest manifest-first fix possible and add tests that prove the path is explicit rather than implicit. If the legacy path dispatches correctly with `defaultKernelPath: null`, record that finding and move to `WS3.4`.
- [ ] `WS3.4` Re-check the Qwen-specific config semantics already present in [`qwen3.json`](../src/config/presets/models/qwen3.json): `queryKeyNorm`, `mropeInterleaved`, `mropeSection`, `partialRotaryFactor`, and `rmsNormWeightOffset`.
- [ ] `WS3.5` Verify whether Qwen linear layers are supposed to use learned Q/K normalization weights, pure in-kernel L2 normalization, or both. Do not assume equivalence with standard-attention RMSNorm.
- [ ] `WS3.6` Audit `linearNormMode` resolution for the refreshed artifacts. If inference from weight shape is ambiguous, move that field to explicit manifest output instead of relying on shape heuristics.
- [ ] `WS3.7` If a local reference environment and source weights are available, run a real deterministic reference comparison against a known-good implementation and isolate divergence to one of the following. If that environment is unavailable, record the skip and proceed to `WS3.8`:
  - manifest/runtime-path setup
  - linear-attention state initialization
  - linear-attention math
  - Qwen-specific normalization semantics
  - tokenizer or prompt-format mismatch
- [ ] `WS3.8` If `WS3.7` identified a specific failure class, strengthen tests so that failure becomes reproducible without relying on a vague "incoherent output" description. If `WS3.7` was skipped, add a focused test stub or TODO anchored to the strongest candidate from `WS3.4` through `WS3.6`.
- [ ] `WS3.9` Re-run a real deterministic Qwen smoke after the fix and record prompt, config, output, and surface.
- [ ] `WS3.10` Stop for human review before updating any catalog or support-matrix claim.

Do not do:

1. Do not force a Qwen-specific kernel path or execution-v0 graph merely because the current conversion config shows `defaultKernelPath: null`.
2. Do not close Qwen from old synthetic report outputs.
3. Do not change linear-attention math until artifact freshness and runtime-path clarity are established first.

## WS4: Gemma Q4K Conversion And Numeric Triage

Status: `blocked`
Blocked on: `WS1`

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

- [ ] `WS4.1` Inspect the refreshed Gemma Q4K manifests and confirm `presetId`, quantization settings, `quantizationInfo` including `layout`, `layerPattern`, `defaultKernelPath`, and execution-v0 state. Note: per Ground Truth item 6, the 1B Q4K model has Class `A` evidence of coherent output on at least one prompt, so the Q4K path is not categorically broken across both Gemma targets.
- [ ] `WS4.2` Verify source checkpoint dtypes for the failing models and record the expected dtype mix. If source checkpoints are not available locally, inspect the corresponding Hugging Face metadata or checkpoint headers where possible and record any remaining access gap explicitly rather than leaving this step implicitly skipped.
- [ ] `WS4.3` Verify shard integrity by sampling hashes against the manifest.
- [ ] `WS4.4` Sample numeric sanity by comparing selected source tensor values with converted bytes and dequantized values.
- [ ] `WS4.5` If any of `WS4.2` through `WS4.4` fails, fix the conversion or artifact issue first and re-run the checks before touching runtime math.
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
