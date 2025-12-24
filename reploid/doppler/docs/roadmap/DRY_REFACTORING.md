# DRY Refactoring Roadmap (Consolidated)

Goal: DRY up the kernel/pipeline code while preserving behavior and performance.

## Scope
- Keep the fused decode path intact.
- Reduce boilerplate in kernels and validation logic.
- Add perf guardrails for submits/allocs/readbacks.
- Treat kernel DRY and cache cleanup as real performance risks, not optional.

## Completed Work
- Perf guardrails wired into submit tracking, buffer allocations, and readback gates (`gpu/perf-guards.ts`, `gpu/submit-tracker.ts`, `gpu/buffer-pool.ts`, `gpu/command-recorder.ts`).
- Readback gating added for debug-only paths and GPU profiling (`inference/pipeline.ts`, `inference/pipeline/layer.ts`, `inference/pipeline/logits.ts`, `inference/kv-cache.ts`, `gpu/profiler.ts`, `gpu/kernels/sample.ts`, `gpu/kernels/check-stop.ts`, `gpu/kernels/cast.ts`).
- Benchmarks capture perf-guard counters (`tests/benchmark/pipeline-benchmark.ts`, `tests/benchmark/types.ts`).
- Sample pipeline cache unified with shared kernel cache (`gpu/kernels/sample.ts`, `gpu/kernels/utils.ts`).
- MoE record variants added and zero-init batching expanded (`gpu/kernels/moe.ts`).
- Dispatch helper adoption and workgroup constants applied to core kernels (`gpu/kernels/softmax.ts`, `gpu/kernels/rmsnorm.ts`, `gpu/kernels/silu.ts`, `gpu/kernels/gelu.ts`, `gpu/kernels/residual.ts`, `gpu/kernels/rope.ts`, `gpu/kernels/gather.ts`, `gpu/kernels/attention.ts`, `gpu/kernels/cast.ts`, `gpu/kernels/dequant.ts`).
- Constants centralized and reused for alignment/workgroup sizing (`gpu/kernels/constants.ts`, `gpu/kernels/matmul.ts`, `gpu/kernels/dequant.ts`, `gpu/kernels/cast.ts`, `gpu/kernels/sample.ts`).
- Kernel base class added with shared run/record dispatch helpers, applied to matmul + attention (`gpu/kernels/kernel-base.ts`, `gpu/kernels/matmul.ts`, `gpu/kernels/attention.ts`).
- Matmul/dequant/cast/check-stop refactors now use shared dispatch and helper logic without changing multi-pass semantics (`gpu/kernels/matmul.ts`, `gpu/kernels/dequant.ts`, `gpu/kernels/cast.ts`, `gpu/kernels/check-stop.ts`).
- Kernel cache unification expanded to shader modules, bind group layouts, and pipeline layouts with purge/stats hooks (`gpu/kernels/utils.ts`).
- Cross-kernel option shapes consolidated via shared types (`gpu/kernels/types.ts` and option interfaces across kernels).
- Type cleanup across pipeline init/context/attention/logits/layer (`inference/pipeline/init.ts`, `inference/pipeline.ts`, `inference/pipeline/layer.ts`, `inference/pipeline/attention.ts`, `inference/pipeline/logits.ts`, `inference/pipeline/types.ts`, `inference/pipeline/debug-utils.ts`).

## Remaining Work
- None.

## Notes
- Multi-pass kernels with explicit staging/copy steps still use direct encoders where needed (sample).

## Validation Checklist
- `doppler test correctness`
- `doppler bench inference --runs 3 --verbose`
- `npm run test:vitest`
