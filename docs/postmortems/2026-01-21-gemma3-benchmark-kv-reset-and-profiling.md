# Gemma 3 Benchmark KV Reset and Profiling Noise

**Date:** 2026-01-21
**Status:** Closed (benchmark correctness fixed; profiling noise reduced)
**Context:** Gemma 3 1B decode profiling in browser (WebGPU)

## Summary

A benchmark correctness bug caused timed runs to reuse warmup KV cache state, inflating attention cost and corrupting decode timing. We fixed KV cache resets and tightened profiling output so logs stay readable while full per-step timings are still saved to disk. This enabled accurate diagnosis of decode bottlenecks (matmul_rmsnorm_fused and attention) using JSON/HTML artifacts.

## Impact

- Timed decode runs started at long-context KV length instead of fresh context.
- Attention timing looked stable across runs, masking real growth behavior.
- Profiling output was too noisy to monitor live execution.

## Root Causes

1) Benchmark harness did not guarantee a clean KV cache before timed runs.
2) SlidingWindowKVCache did not reset totalTokensSeen on clear(), so KV metadata persisted even after reset.

## Fixes

- Reset pipeline before each warmup and timed run in the benchmark loop.
- Reset sliding-window totalTokensSeen in KV cache clear().

## Tooling Improvements

- Capture per-step GPU profile timings into JSON artifacts for plotting.
- Add decode kernel timing charts and attention crossover summary to HTML report.
- Add decode per-token throughput chart (tok/s) in HTML report.
- Reduce profiler log noise with logEveryDecodeSteps while keeping full artifacts.

## Findings from the corrected runs

- Attention cost grows with KV length and eventually overtakes matmul in Q4K runs.
- Matmul timing stays relatively flat while attention rises with context, which helped confirm the KV reset bug and validate the per-step profiler data.
- Q4K decode path is slower largely due to kernel path fallback to generic matmul for 1152-wide GEMV.
- matmul_rmsnorm_fused is the FFN down_proj + post-FFN norm + residual and is the dominant kernel in decode.
- Buffer reuse is not the primary bottleneck in these runs; decode is GEMV and attention dominated.

## What We Learned

- Benchmark correctness must be validated before interpreting GPU timing data.
- Persisting per-step profiles in artifacts is critical; console logs can be sparse.
- Kernel path selection can dominate quantization performance outcomes.

## Follow-ups

- Add op-level labels to profiling output to map matmul_rmsnorm_fused to specific FFN ops.
- Evaluate multi-token decode and tuning for GEMV kernels.
- Investigate non-aligned Q4K GEMV kernel paths for Gemma 3 hidden size 1152.
