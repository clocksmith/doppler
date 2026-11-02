# Qwen3.6 35B startup optimization, Jupiter

This is retained browser/WebGPU tuning evidence, not a claim-grade overall
Doppler-versus-Pooled result. The final frozen source matches the checked-in
runtime tree, including its compatibility symlink. No model weights, arithmetic,
KV precision, sampling, or numerical acceptance thresholds changed.

The final run observed Doppler ready in **79.00 s**, versus Pooled **93.07 s**.
The clean Doppler baseline was **240.43 s**: a 67.1% reduction in observed opening
time. Each source has only one clean startup observation. These are fresh browser
profiles with uncontrolled OS file caches, not cold-install or startup-tail
qualification. Startup includes the configured warm generation; costs were not
moved beyond readiness.

## Implementation

- Complete-byte SHA-256 uses WebCrypto when available. Incremental verification
  retains its bounded implementation; native errors propagate.
- Full-shard loads verify the exact bytes returned to their consumer, avoiding
  the old separate verification pass and subsequent read.
- The experimental 35B profile selects
  `loading.shardCache.verificationBufferBytes = 67108864`. Eligible HTTP range
  sources verify one complete buffer at a time per storage context. Larger
  sources retain incremental verification. The default remains zero.

The buffer budget covers the verifier's input workspace. Native crypto can make
an additional copy; this is not a process-wide memory bound. Corrupt, truncated,
or unsupported inputs fail explicitly. Failed verifications release the queue
and remain retryable.

## Final alternating requests

Run: `f7-profile-startup-tail-symlinks`. Ten measured alternating pairs per
throughput prompt, after one warmup per engine/prompt; both engines resident.
No outlier removal. Values below are milliseconds; lower is better. Differences
are observed diagnostics and do not establish a mechanically admitted win.

| Metric | Doppler | Pooled | Doppler minus Pooled |
| --- | ---: | ---: | ---: |
| Opening through readiness (one observation) | 78996.10 | 93066.60 | -14070.50 |
| Short prefill, mean | 137.00 | 124.79 | +12.21 |
| Short first token, mean | 141.35 | 140.02 | +1.33 |
| Short inter-token interval, mean | 13.603 | 13.957 | -0.354 |
| Short 128-token request, mean | 1869.63 | 1912.85 | -43.22 |
| Short request p95 | 1895.62 | 1936.17 | -40.55 |
| Long prefill, mean | 4224.62 | 3423.19 | +801.43 |
| Long first token, mean | 4229.95 | 3440.08 | +789.87 |
| Long inter-token interval, mean | 14.202 | 14.589 | -0.387 |
| Long 128-token request, mean | 6034.29 | 5293.12 | +741.17 |
| Long request p95 | 6080.54 | 5423.15 | +657.39 |

These prompts contain 28 and 2287 input tokens respectively. Inter-token interval
is `(last delivery - first delivery) / 127`, excluding first-token latency.
Percentiles use interpolated index `(n - 1) * p`; ten observations do not qualify
production tails. `ci95` in the JSON describes the mean, not percentile uncertainty.

The unchanged frozen harness's equal-prompt paired request estimator is 1.05564
Doppler/Pooled, with 95% bootstrap interval [1.04822, 1.06207]. Across these two
prompts Doppler remains slower overall. Long throughput outputs hit the 128-token
cap, so these are not complete-long-answer measurements. Four separate frozen
admission tasks reach EOS and pass; they do not establish broad quality parity.

## Scope and correctness

Jupiter: AMD RDNA 3, Linux, HeadlessChrome 146, physical WebGPU; reported features
include shader-f16, subgroups, and timestamp-query. This evidence does not qualify
Node, other browsers, GPUs, drivers, or models. Every timed run held the shared
`/run/media/x/models/doe-gpu-timing.lock`.

Model: `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`; Doppler RDRR source
`deco-evidence/doppler-35b-native/rdrr-q4_0-e12da725`. Manifest and artifact
identities, executed options, browser identity, and resolved runtime are retained
in ready receipts. Doppler uses F32 activations/KV; Pooled retains its native
default flash/F16 KV configuration. No speculation. Common executed weights and
the independent numerical gate remain unqualified. This report does not replace
the [numerical investigation](../qwen36-35b-pooled-numerics-20261031/README.md).

All 360 outputs across the six retained runs match the first observed output
for their engine/task exactly (prompt IDs, token IDs, and text). All frozen
quality assertions pass, with no recorded GPU errors. This establishes
continuation under the loading changes; it is not independent logits parity.

## Retained experiments and snapshot correction

| Run | Change | Doppler ready, seconds | Pooled ready, seconds |
| --- | --- | ---: | ---: |
| s19 | Clean committed baseline `5ba15838e` | 240.43 | 97.85 |
| s20 | Native complete-buffer SHA-256 | 184.49 | 93.34 |
| s21 | Verify full-shard returned bytes once | 135.62 | 89.02 |
| s22 | Explicit 64 MiB range-verification budget | 88.25 | 93.79 |
| f6 | Budget selected by runtime profile | 88.00 | 90.00 |
| f7 | Correctly preserved source symlink | 79.00 | 93.07 |

The snapshot copier expanded `src/inference/config -> ../config` into a directory
for s20, s21, s22, and f6. That directory contained a stale copy after applying
subsequent patches. Those runs are retained diagnostics, not final-source
replications. f7 was repeated from a POSIX snapshot preserving the tracked link;
its `src/` identity exactly matches the working tree. The external model volume
does not support symlinks, so source archives preserve their metadata in tar.

The intermediate observations suggest where loading time was spent, but their
snapshot differences and uncontrolled caches prevent attributing precise causal
percentages to individual changes. The clean s19/f7 comparison supports the
combined optimization; it does not establish performance variance by itself.

## Evidence and reproduction

- [Measurements](measurements.json): scalar samples, statistics, continuation
  fingerprints, source identity, and snapshot limitations.
- [Raw-file index](retained-files.json): SHA-256 bindings for receipts, frozen
  contracts, ready records, every attempt, and the final harness summary.
- [Source archives](source-archives.json): SHA-256 bindings for all six snapshots.

External paths are relative to `/run/media/x/models/`. Raw runs live under
`deco-evidence/doppler-35b-perf/`; each includes frozen harness code in
`executed-source/qwen-economics/`. Final source archive is
`deco-evidence/doppler-35b-startup-20261101/source-archives/f7-source.tar.gz`.
The selected `src/` tree digest is
`00aa767dedb8e6b02963a0c74115bd316298b52529ebbc5149c8485b140c3b08`;
its construction is recorded in measurements. Execution records bind the
overlays, source paths, model paths, and contract. Reproduction must extract the
source archive on a filesystem supporting symlinks and use a new output directory;
do not reuse a runner that deletes existing evidence.

The final config is
`deco-evidence/doppler-35b-startup-20261101/final-profile-symlinks.json`.
Its frozen runner was invoked under the shared GPU lock with a bounded timeout.
Earlier profile diagnostics did not capture all unbatched prefill dispatches;
their decode-only GPU timings must not be used to classify the long-prefill wall.
Long prefill is the next unresolved performance owner.

## Development acceptance

Passed: SHA-256 incremental/native/view/failure tests; BLAKE3 vectors; artifact
storage integrity, range, bounded-concurrency and retry tests; shared harness
budget forwarding; `source:style:check`, `config:single-source:check`,
`typecheck:source`, `onboarding:check`, and `git diff --check`.
These checks do not establish full release or multi-device qualification.

Component: `doppler.runtime-source.storage`, `doppler.runtime-source.config`,
`doppler.runtime-source.tooling`, `doppler.tests`, `doppler.docs`.
Intent: preserved.
Acceptance evidence: focused commands above and the bound browser receipts.
Boundary effects: none.
