# Qwen3.6 35B F32 prefill tile, Jupiter

The registered F32 prefill attention tile now admits sixteen queries per
512-invocation workgroup as well as the original eight-query/256-invocation
variant. Queries reuse the same 16-key KV tile. Per-query arithmetic, subgroup
partitioning, mask handling, weights and F32 activation/KV precision are unchanged.
The original variant remains available.

The experimental `profiles/qwen3-6-35b-a3b-q4_0-candidate` selects the measured
sixteen-query variant with 2048-token prefill chunks. This is a Jupiter tuning
profile requiring 512-invocation workgroups and `subgroup_id`; it is not
qualification for another GPU, browser or Node. Decode and speculation are unchanged.
No model artifact, catalog support or release claim is promoted.

## Fixed paired requests

Ten measured alternating pairs per prompt after one warmup, with both engines
resident. The short prompt has **28 input tokens**, the longer prompt **2287**;
both generate **128 output tokens** and stop at the cap. They do not measure
naturally completed answers. No outliers are removed. Four frozen admission
tasks repeat eleven times per engine: all 88 pass, with no GPU errors in any
of the 132 total attempts. All ten pairs were retained before analysis.

| Mean | Doppler | Pooled |
| --- | ---: | ---: |
| 28-input-token first output token | 139.77 ms | 140.42 ms |
| 28-input + 128-output-token request | 1865.18 ms | 1908.57 ms |
| 2287-input-token first output token | 3349.50 ms | 3412.73 ms |
| 2287-input + 128-output-token request | 5156.67 ms | 5267.74 ms |
| Short inter-token interval | 13.5859 ms | 13.9224 ms |
| Longer inter-token interval | 14.2297 ms | 14.6064 ms |

Paired Doppler/Pooled request ratios and 95% percentile bootstrap intervals:
short **0.977263 [0.972719, 0.981128]**; longer **0.978916
[0.976190, 0.981796]**. The equal-prompt estimator is **0.978089
[0.975514, 0.980506]**. These describe these fixed prompts, not a distribution
of user demand. Longer first-token ratio is 0.981462 [0.978363, 0.985150];
short first-token ratio is 0.994531 [0.965282, 1.025560], so no short first-token
win is established. Ten samples do not qualify production latency tails.

Opening through readiness was observed once: Doppler 86.710 s, Pooled 91.265 s.
Fresh browser profiles do not control OS file caches; this is not a cold-install
or startup-distribution claim.

## Native prefill timing correction

The bound adapter calls Pooled `prefillTokens(promptIds.slice(0, -1))`, then
processes the last prompt token with `forwardTokenIds` to produce its first
output. Doppler's `prefillTimeMs` covers `_prefillPromptToLogits` for the full
prompt. Thus native counters cover 27 versus 28 input tokens, or 2286 versus
2287, and different logit boundaries. They are not directly comparable prefill
latencies. This applies to previous reports using the same adapter.

The retained analysis checks the exact adapter SHA-256 before assigning these
boundaries. It preserves native counters and excludes them from paired
cross-engine estimators. First-token delivery and 128-token request completion
remain the matched timing boundaries; the shared inputs and output budgets did
not change.

## Correctness and retained configuration

Nine physical browser GPU fixtures are bit-identical to the original kernel.
They include query tails, prefixes, sliding windows, bidirectional spans,
fully masked rows and the model's query/KV head grouping. Independent CPU
samples remain within the unchanged 0.0001 tolerance (worst below 7e-9).
An instrumented component experiment measures the 2048-token attention shape
at 49.382 ms original versus 34.499 ms candidate; these timestamps are separate
from the primary uninstrumented requests.

Full-model untimed probes compare fifteen saved prefixes, including chunk
boundaries and both actual workload prompts. **All 3,724,800 logits match
exactly**, with identical greedy tokens; the absolute 0.001/relative-zero gate
is unchanged. The reference is the prior F32 implementation, not an independent
full-model oracle.

A separate physical run consumes the checked-in profile with no kernel-path
or chunk-size override. Its resolved runtime equals the primary run's effective
runtime, all six outputs match, admission tasks pass and no GPU errors occur.
Its consumed `src/` tree equals the final source tree, including symlink identity.

Pooled retains its native executed weight materializations and F16 KV while
Doppler remains F32. Generated tokens match across engines for five tasks,
but differ on the longer throughput task. Frozen admission tests are not broad
quality equivalence. Common-weight/independent numerical, generated-position
router and best speculative qualification remain open. These product-format
measurements do not establish an across-the-board matched-quality engine ranking.

## Rejected experiments

The single-chunk 4096 policy did not improve measured prefill. Named temporary
vectors in the projection and attention shaders did not establish a sustained
long-prefill gain; those source edits were restored. Their receipts remain.
An explicit F16 KV/flash lane stopped during warm generation when whole-token
replay rejected a changing bind-group layout. That failed lane is retained and
is not used by this profile; its guard was not bypassed.

## Evidence and development checks

[Measurements](measurements.json), [raw-file bindings](retained-files.json), and
[source bindings](source-files.json) retain successful, failed and rejected runs.
External evidence root: `/run/media/x/deco-backup/doppler-perf-20261102`.
The fixed primary run is `attention-tile16-pooled-final`; numerical probes are
`attention-tile16-numerical`; direct profile consumption is `retained-profile-check`.
Source snapshots preserve the tracked `src/inference/config` symlink.

Focused checks pass: `attention-online-constants.test.js` (immediate/recorded
geometry), `attention-f32-prefill-selection.test.js` (pinned specialization and tails), `kernels:check`, `onboarding:check`, `typecheck:source`,
`source:style:check`, refreshed `runtime:closure:check`, and `git diff --check`.
No broad release or device matrix was run.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.runtime-source.config`,
`doppler.tests`, `doppler` (retained evidence).
Intent: preserved.
Acceptance evidence: commands and bound physical receipts above.
Boundary effects: experimental profile geometry/chunk policy; no model arithmetic or artifact changes.
