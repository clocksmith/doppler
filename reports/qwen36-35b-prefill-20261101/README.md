# Qwen3.6 35B prefill attribution on Jupiter

This report closes through s28. The later
[native MoE rewrite](../qwen36-35b-native-moe-registers-20261101/README.md)
records the accepted implementation and its separate evidence.

The long-prompt deficit is primarily GPU execution in the measured Doppler
path. A diagnostic observer captured **8472 compute passes**, totaling
**3902.69 ms**, across a **4023.20 ms** GPU timestamp span. Host first-token
latency was **4034.00 ms** and reported prefill was **4026.30 ms**. This is
timestamp-instrumented diagnostic evidence, not a throughput comparison.

| Observed pipeline group | Passes | GPU milliseconds |
| --- | ---: | ---: |
| Routed MoE Q4_0 gate/up | 360 | 986.42 |
| Standalone Q4_0 tiled prefill projections | 684 | 795.73 |
| Attention, gate and output projection combined passes | 90 | 439.24 |
| Routed MoE Q4_0 down | 315 | 426.88 |
| Linear-attention recurrence | 270 | 271.35 |
| Standalone Q8_0 prefill projections | 666 | 227.59 |
| Standalone Q5_0 prefill projections | 540 | 157.42 |

Combined passes contain multiple pipelines; their entire time must not be
assigned to just one operation. The complete list and API deltas are in
[measurements.json](measurements.json). The first-delivery observation contains
8125 queue submissions and 8742 direct plus 720 indirect dispatches. CPU and
submission work exists, but GPU pass durations already account for most of
this instrumented first-token wall. Concurrent completion-wait counters overlap;
their large summed values are not additional latency.

The existing generation profiler did not capture the immediate path used by
token-chunked prefill. Earlier partial or decode-only profiles cannot establish
the complete prefill attribution. This observer wraps `beginComputePass`, adds
timestamp writes to each existing pass, and records pipelines/dispatch geometry.
It never splits passes or changes dispatches. A pre-existing timestamp owner or
exhausted fixed query budget fails explicitly. Collection occurs after the
request's recorded completion; resources and prototype hooks are cleaned up.

Scope: one warmup long request followed by one diagnostic request, 2287 input
tokens and 128 output tokens; frozen source from `4d65903d1`, the unchanged
experimental candidate profile, 256-token chunks, F32 activations/KV, no
speculation. Whole-token decode replay stays enabled. Both outputs exactly match
the final [startup experiment](../qwen36-35b-startup-20261101/README.md), with
no recorded GPU errors. This is continuation evidence, not independent numerical
qualification, complete-long-answer measurement, or a Pooled comparison.

Environment: Jupiter, Linux, AMD RDNA 3 physical WebGPU, HeadlessChrome 146.
The shared GPU timing lock was held for the run. No Node or other-device claim.
The timestamp budget was 2048 queries per block, at most 32768 passes; nine blocks
were used. A simultaneous CPU sampling profile is retained and is diagnostic.

Raw receipts, CPU profile, and the executed observer/harness are SHA-256 bound by
[retained-files.json](retained-files.json), relative to `/run/media/x/models/`.
Config:
`deco-evidence/doppler-35b-startup-20261101/immediate-prefill-profile.json`.
Runtime source identity and preserved source archive are bound in the startup
report. The unchanged routing audit was run before considering further kernel
work; its output is retained externally as
`deco-evidence/doppler-35b-startup-20261101/routing-audit.json`.

## Follow-up experiments

Each screen uses one warmup and three measured alternating pairs per throughput
prompt, plus four admission tasks. Both sessions are Doppler with the same model
and F32 settings. These are small tuning screens, not Pooled comparisons.

| Experiment | Baseline long prefill, ms | Candidate, ms | Disposition |
| --- | ---: | ---: | --- |
| s23: 512-token chunks | 4179.87 | 4005.93 | Retained diagnostic |
| s24: 1024-token chunks | 4207.70 | 3856.37 | Retained diagnostic |
| s25: fixed group-loop bounds | 4178.73 | 4209.37 | Rejected; shader restored |
| s26: named private accumulators | 4207.37 | 4131.70 | Not accepted; resident-session order confounds the small difference |
| s27: 2048-token chunks | invalid | invalid | GPU dispatch failure; all run timings inadmissible |
| s28: repaired convolution grid, 2048-token chunks | 4175.13 | 3784.30 | Correctness repair passes; chunk policy still a candidate |

The first resident session also has a decode advantage in these own-engine
screens, including when order reverses. Do not assign that difference to a
prefill-only shader change. No MoE shader experiment was retained at s28.
The larger private-array rewrite was then an untested parked draft.
An intermediate named-accumulator edit failed compilation; its failure is kept,
and the corrected version passed before the model screen.

[Experiments](experiments.json) retain samples, failures and dispositions.
[Raw artifacts](experiment-files.json) and [source archives/patches](experiment-sources.json)
bind the executed and rejected variants. The earlier chunk-only summary and file
index are also retained; they cover s23/s24 only.

## Convolution correctness repair

The 2048-token screen exposed the first invalid operation: parallel convolution
flattened `2048 * 8192 / 256 = 65536` workgroups onto X, beyond the device's
65535-per-axis limit. The failing warmup returned GPU validation errors; its
apparently shorter latency is invalid. The following request stopped on those
errors. No quality or performance credit is given to this run.

The repair maps channels to X and tokens to Y. The failed shape now dispatches
`[32, 2048, 1]`. Both immediate and recorded paths consume the same validated grid,
with admission before output/scratch allocation. The existing dispatch-limit
validator remains the owner of that check. Unsafe token counts fail explicitly.
The generated shader and its canonical codegen patch change together; both
entry-point digests are regenerated. No weight, precision, normalization,
sampling, or model metadata changes.

Physical browser/WebGPU checks compare every output bit with the serial GPU
convolution, CPU samples, and final causal history. Cases include short inputs,
nonzero incoming state, packed QKV/Z stride, channel tails, and both 2047/2048
tokens with 8192 channels. All pass without GPU errors; worst sampled CPU error
is below `6.4e-10`. This is component evidence, not Node model qualification.

The repaired original model screen (s28) completes all 48 attempts, with all four
quality tasks passing and every output identical to its 256-token control.
Long request means are 5620.57 ms candidate versus 6002.93 ms control. Short
request means are 1917.70 versus 1838.37 ms; retain that observed difference.
These timings do not establish a Pooled win or latency-tail qualification.
The candidate profile still selects 256-token chunks while full-vocabulary
checks around the proposed chunk boundaries run separately.

Frozen repair source archive:
`deco-evidence/doppler-35b-startup-20261101/conv-grid-fix/source.tar.gz`.
The structured first-failure diagnosis is retained under
`conv-dispatch-repro/diagnosis.json` in the same evidence root.

Development checks passed: `linear-attention-conv-grid.test.js` (GPU mocked,
immediate/recorded coverage, rejection before allocation), existing
`linear-attention-allocation-cleanup.test.js`, the shared physical GPU reference
fixture, `kernels:check`, `typecheck:source`, `source:style:check`,
`runtime:closure:check`, `package:smoke`, and `git diff --check`.
No blanket GPU or release suite was run.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.tests`,
`doppler.repository-tooling` (external diagnostic adapter), `doppler` (evidence).
Intent: preserved.
Acceptance evidence: completed timestamp and repair receipts, physical component
reference, unchanged repair-run outputs, and focused checks above.
Boundary effects: none.
