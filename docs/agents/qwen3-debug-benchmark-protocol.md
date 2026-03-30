# Qwen 3.5 Debug / Benchmark Protocol

Referenced by: `doppler-debug`, `doppler-bench`, `doppler-perf`

Use this addendum when a Qwen 3.5 run is being used to judge correctness, performance, or compare-lane promotion.

## Scope

Applies to:

- `qwen-3-5-0-8b-q4k-ehaf16`
- `qwen-3-5-2b-q4k-ehaf16`

Primary risk areas:

- published artifact freshness vs local artifact freshness
- full-attention layers vs linear-attention / delta-net layers
- learned `q_norm` / `k_norm` expectations on full-attention layers only
- compare-lane evidence that looks informative but is not claimable

## Core Rules

1. Prove the active artifact source first.
   - Compare runs default to the declared source in [compare-engines.config.json](../../benchmarks/vendors/compare-engines.config.json).
   - Do not assume a local manifest fix is present in the published quickstart artifact.
2. Treat correctness as a gate before reading perf.
   - A mismatch compare artifact is diagnostic evidence, not a claimable Qwen benchmark result.
3. Separate contract failures from kernel failures.
   - Qwen regressions often come from manifest, execution-graph, or harness-shape drift before they come from kernel math.
4. Debug lanes are not perf lanes.
   - Do not use `f32`, `http`, layer-probe, or investigation-only profiles as claimable performance evidence for the warmed Chromium / OPFS compare lane.
5. Full attention and linear attention are different subsystems.
   - Standard attention layers can use learned `self_attn.q_norm` / `self_attn.k_norm`.
   - Linear-attention layers use the Qwen delta-net path and should not inherit standard-attention Q/K-norm assumptions.

## First Pass Order

1. Active artifact diff
   - Record `modelId`, `modelUrl`, `manifestSource`, `manifestSha256`, and source kind (`quickstart-registry` vs local).
   - If the run is browser-based, confirm whether warm OPFS state is part of the intended lane.
2. One raw correctness lane
   - Use a deterministic prompt and greedy sampling first.
   - Record the exact generated text, token prefix match, and first mismatch token index before reading throughput.
3. Resolved kernel path and dtype dump
   - Dump the resolved prefill/decode kernel path for `linear_qkv_proj`, `linear_z_proj`, `linear_out_proj`, attention, and FFN projections.
   - Dump the actual loaded weight/materialization dtype for the same ops.
   - Do not touch kernels until the selected phase path matches the loaded weight view.
4. Layer-family split
   - Read the manifest `layerPattern` and separate full-attention layers from linear-attention layers.
   - Only expect learned `q_norm` / `k_norm` tensors on the full-attention layers.
5. Operator proof before kernel edits
   - Prefer operator diagnostics, targeted probes, or microbenches over repeated full compare reruns.
   - For Q4 prefill work, a small projection or attention microbench is a faster truth source than another full end-to-end compare loop.

## Decode Pass Order

When Qwen decode is the suspected wall, start here before touching kernels:

1. Dump the decode contract first.
   - Record `decodeMode`, `batchGuardReason`, `disableMultiTokenDecode`, active speculation state, resolved decode kernel path, and actual loaded decode weight/materialization dtype.
2. Separate compute from orchestration.
   - Use `decodeRecordMs`, `decodeSubmitWaitMs`, `decodeReadbackWaitMs`, `singleTokenReadbackWaitMs`, and `singleTokenOrchestrationMs` to decide whether the bottleneck is GPU math or command/readback overhead.
3. Treat prefill and decode as different policy surfaces.
   - Qwen may need different kernel-path/rule decisions for prefill and decode; do not assume a prefill win or loss says anything about decode.
4. Only then try decode kernel or rule changes.
   - If submit/readback dominates, prefer recorder/readback/orchestration fixes over math-kernel edits.

## Qwen-Specific Checks

- `attention.queryKeyNorm` belongs to standard attention. Linear-attention dispatch must not silently inherit it.
- `rope.mropeInterleaved`, `rope.mropeSection`, `partialRotaryFactor`, and `rmsNormWeightOffset` must come from manifest/runtime contract, not runtime guesses.
- Qwen linear-attention perf work should start from the real compare lane and then narrow to operator timing. Do not begin from the recurrent decode kernel unless prefill evidence points there.
- When using an explicit local `file://` model source in browser relay, verify the run is not silently reading a warm OPFS artifact unless that is the intended lane.

Useful references:

- [model-failure-action-plan.md](../model-failure-action-plan.md)
- [qwen3-5-linear-attn-debug.json](../../src/config/runtime/model/qwen3-5-linear-attn-debug.json)
- [qwen3-5-layer-probe.json](../../src/config/runtime/model/qwen3-5-layer-probe.json)
- [qwen3-bench-q4k-batched.json](../../src/config/runtime/experiments/bench/qwen3-bench-q4k-batched.json)
- [hf_qwen35_linear_attn_debug.py](../../src/debug/reference/hf_qwen35_linear_attn_debug.py)

## Promotion Gate

- `qwen-3-5-0-8b-q4k-ehaf16` may remain `performance_comparable` only if there is committed correctness-clean compare evidence for the shared compare workload.
- `qwen-3-5-2b-q4k-ehaf16` stays `capability_only` until compare config, compare-lane reason, and correctness-clean evidence are updated together.
- Do not publish README or chart claims from Qwen mismatch artifacts or debug-only lanes.

Minimum claimable evidence for a Qwen compare lane:

- one committed compare fixture with `correctness.status: "match"`
- `exactMatch: true`
- `firstMismatchTokenIndex: -1`
- exact workload ID and reproduction command
- browser / GPU / runtime metadata
- explicit artifact-source identity
- parity compare decode semantics with compare-managed self-speculation disabled
