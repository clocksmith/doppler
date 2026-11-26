# Debug Playbook

How to read the results of the inference debug ladder for inference regressions,
incoherent output, and conversion-vs-runtime triage.

The ladder itself (failure classes, trusted reference, boundary order, quantized
control, conversion completion) is canonical in
[Inference Debug Protocol](agents/debug-protocol.md). This page adds the
interpretation for each step. Its goal is the shortest path to the first
divergent boundary.

## 1. Classification

Use the protocol's failure classes. Do not mix categories until the current one
is either confirmed or ruled out.

## 2. Trusted reference

In addition to the protocol's capture list, record the deterministic sampling
tuple. Without the full set you do not yet have a stable comparison target.

## 3. Reading boundary diffs

Compare the same token in the protocol's boundary order and stop at the first
divergence:

- divergence before RoPE usually points to weight loading, dequantization, tensor layout, or matmul
- divergence after RoPE but before attention output usually points to RoPE application or masking
- divergence only after attention output points to residual/FFN/downstream path

## 4. Reading the quantized-model control split

When a quantized artifact is bad, do not edit quantized kernels first. Run the
protocol's F16 or source-precision control on the same prompt and compare the
same early boundaries:

- F16/source-precision good, quantized bad: quantization, dequantization, runtime dequant layout, or quantized matmul path
- F16/source-precision bad, quantized bad: shared conversion/layout/runtime issue
- embeddings + post-input-norm match but Q/K/V diverge: projection weight path, dequant path, or matmul dispatch

This control experiment is required before claiming a Q4K-specific or runtime-wide bug.

## 5. Stop harness churn when early boundaries match

If token IDs match, stop changing prompt formatting.

If embeddings match, stop changing tokenization theories.

If post-input-norm matches, stop changing RMSNorm theories.

Move to the next unresolved boundary instead of reopening solved ones.

## 6. Conversion triage is separate

For fresh conversions:

1. verify source dtypes
2. verify manifest quantization/default kernel path
3. verify sampled shard hashes
4. verify sampled tensor bytes against source
5. then and only then classify runtime divergence

Conversion completion is fail-closed; see the protocol's conversion-status step.

## 7. Add the smallest permanent probe

Prefer:

- existing trace categories
- config-driven probes
- one permanent probe-stage extension

Avoid:

- throwaway `console.*`
- broad benchmark rewrites before classification
- speculative config changes without a boundary diff

## 8. Required deliverables for a serious debug pass

- failing command/config
- trusted reference values
- first divergent boundary
- classification of likely owner (`tokenization`, `conversion`, `runtime`, `surface`)
- next binary split or fix

Use [debug-investigation-template.md](debug-investigation-template.md) when the investigation spans multiple iterations or agents.

## GPU phase timing diagnostics

`CommandRecorder` accepts `profilePhase: 'prefill' | 'decode' | null` as explicit
observation metadata. `setProfilePhase()` attributes subsequent passes; labels do
not select phases. Generation recorders bind their phase at creation. Resolve once
with `resolveProfileTimings({ groupByPhase: true, requireComplete: true })` for
phase-to-kernel millisecond totals; unscoped passes are `unattributed`. Omitting
`groupByPhase` preserves the existing per-kernel aggregate.

Timestamp profiling splits compute passes and reads query buffers. Keep these
instrumented diagnostics separate from ordinary complete-request latency and
throughput samples. Recorder stats expose the requested timing class, timestamp
availability and timestamped/untimestamped pass counts, including after cleanup.
Strict resolution rejects unavailable timestamps or incomplete pass coverage;
partial observations cannot establish a complete phase cost. Changing observation
phase does not add passes, submits or waits to an ordinary recorder.

Generation profiling resolves each recorded prefill/decode segment with complete
coverage required. Unavailable timestamps or exhausted queries fail the diagnostic
request before a partial sum can enter the phase GPU totals. This opt-in failure
does not affect ordinary requests. GPU pass totals exclude host orchestration,
copies and other work outside the recorder; they are not complete-request latency.
