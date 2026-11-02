# Qwen3.6-35B exact GPU feedback replay, Jupiter

The experimental `profiles/qwen3-6-35b-a3b-q4_0-candidate` queues greedy decode
steps whose sampled GPU token feeds the next step directly. Per-token readbacks
start after each submission, so tokens stream while later queued steps run.
The window ramps from ordinary token delivery to a configured ceiling of 32;
existing model/device caps and declared program positions still bound it.
Weights, WGSL arithmetic, sampler rules, F32 activations/KV, the sixteen-query
prefill attention tile and 2048-token chunks are unchanged from the parent.

Each window owns a recurrent/convolution checkpoint through publication. EOS,
stop sequences, abort, consumer return and delivery errors drain queued work,
then restore and replay the published prefix when needed. These costs remain in
request wall time. A window can delay cancellation while queued GPU work drains.
Other models, devices and Node are not qualified by this browser run.

## Fixed paired requests

Ten measured alternating pairs per condition, after one warmup with both engines
resident. Every throughput request emits 128 tokens and ends at the output cap;
these are not naturally completed coding jobs. Inputs contain 28 or 2287 tokens.
Pooled's frozen native room preset uses its real MTP head at draft depth 3, with
prompt lookup tested separately off and on. Every setting remains visible.

| Input tokens / Pooled lookup | Doppler mean request | Pooled mean request | Paired D/P ratio [95% interval] |
| --- | ---: | ---: | ---: |
| 28 / off | 1679.21 ms | 1719.27 ms | 0.976681 [0.970940, 0.983275] |
| 2287 / off | 4978.55 ms | 5261.30 ms | 0.946259 [0.943514, 0.949024] |
| 28 / on | 1677.37 ms | 1793.44 ms | 0.935266 [0.930447, 0.940408] |
| 2287 / on | 4977.07 ms | 5360.45 ms | 0.928477 [0.926429, 0.930231] |

All four complete-request intervals are below one. They describe these fixed
conditions, not a population of user demand. No individual outliers were removed.
All 176 attempts completed without GPU errors; all 88 frozen admission attempts
passed. Those admission tasks do not establish broad task-quality equivalence.
Within each engine, every greedy output matches its retained plain baseline.
Cross-engine output equality is recorded per condition in the measurements;
the longer prompt differs across engines.

Requests start after state reset and its GPU fence. Completion includes GPU
settlement and application text delivery. Pooled's native prefill counter omits
the final prompt token; Doppler's includes it. Those counters have different
logit boundaries and are excluded from cross-engine estimators. First-token
and complete-request measurements remain separately recorded. Opening was
observed once per engine without controlling OS file caches; no cold-install
or startup-distribution claim is made. Token-gap statistics are within-request
observations, not production latency-tail qualification.

## Exact transport correctness

The source-bound physical gate compares ordinary parent replay with the final
implementation: 2,483,200 actual sampling-logit values match byte-for-byte across
ten probes, comprising seven distinct input/output-cap conditions and repeated
recovery controls. Thirty complete recurrent/convolution states and ten live KV
prefixes also match at every probe. Six short-prompt caps, the 2287-input/128-output
request, cancellation and an injected callback failure are covered. The last
two retain exactly the state and logits of their six-token control.

The reference is Doppler's parent implementation, not an independent full-model
oracle. Native Pooled executed weight materializations and F16 KV differ from
Doppler's F32 configuration. Common-weight independent numerical, generated-position
router and broad matched-quality qualification remain open. No model artifact,
catalog support, commercial service or economics claim is promoted.

## Retained experiments and capture integrity

Four-row projection grouping preserved exact logits but did not improve the
short request and was restored. Direct operator geometry screens remain
instrumented diagnostics. The first GPU batch pilot entered the older recording
batch loop; the new routing test prevents that path from owning prepared replay.
Initialization rejections, the empty-task audit, and expected-abort classification
failures remain retained. Task selection now rejects empty or incomplete requests
before browser startup; injected delivery failures keep their exact diagnostics
and require explicit matching controls.

The first full timing capture overlapped host CPU verification around 19:42:11–14
UTC. The entire run remains retained and is excluded from the claim. Its clean
replacement uses the same source, policies and fixed ten-pair count; no samples
were selectively removed. Verification and identity hashing run after timing.

## Evidence and development checks

[Measurements](measurements.json), [exact state/logit gate](numerical-state-gate.json),
[source bindings](source-files.json), [raw-file bindings](retained-files.json), and
[development checks](development-checks.json) bind the results. External evidence
root: `/run/media/x/deco-backup/doppler-spec-20261102`. The accepted request run is
`gpu-stream32-pooled-qualified`; the numerical run is
`gpu-stream32-final-numerical`. Raw source snapshots preserve the tracked
`src/inference/config` symlink. The raw GGUF and RDRR manifest are byte-hashed;
Pooled engine/room/harness sources are independently bound.

Nine focused behavioral checks and the type, source style, config, onboarding,
inference-boundary, export-parity and Capsule-closure checks pass. The source
architecture check still reports the same two pre-existing file-size ceilings
as unchanged parent source (`prefill-runtime.js` and
`artifact-storage-context.js`). Those files and policy ceilings are untouched.
No broad release or device matrix was run.

Component: `doppler.runtime-source.inference.pipelines.text`, `doppler.tests`,
`doppler.repository-tooling`, `doppler.docs`.
Intent: preserved.
Acceptance evidence: bound physical receipts and focused checks above.
Boundary effects: prepared greedy replay transport and experimental 35B window
policy; model weights, shader math and application sampling semantics unchanged.
