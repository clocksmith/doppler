# Gemma 4 E2B INT4 PLE — Perf Loop Notes

Model: `gemma-4-e2b-it-q4k-ehf16-af32-int4ple`
Machine: Apple M3 MacBook Air, 24 GB unified memory, Metal 3, f16 + subgroups
Loop owner: doppler-perf skill, self-paced (5 iters, 2026-04-26)
Status: **stopped** — handoff to follow-up.

## Handoff TL;DR

**Baseline**: 10.6 tok/s decode, 80 tok/s prefill, 792 ms TTFT, 7.5 GB peak. Manifest already routes the fast head256/head512 attention kernels — that win is already shipped.

**The wall is FFN matmul GPU compute.** Per-kernel profile (iter 2, see `iter-002-profile.json`):
- FFN ffn_down + ffn_gate + ffn_up = **73.6% of prefill GPU**, **57.2% of decode GPU**
- lm_head decode = 14.3% of decode GPU
- Attention = 4.5% of prefill, 4.0% of decode (already optimized)
- PLE prep is <8% — initial hypothesis was wrong

**The single biggest win is blocked by one bug.** All q4k-fusion kernels (`fused_ffn_q4k_f16`, `fused_matmul_q4_widetile*`, `fused_rmsnorm_q4_widetile`) sit unrouted in `src/gpu/kernels/`. They require `retainQ4KMaterialization: true`. That flag is force-disabled by capability transform on Apple Metal:

> `src/rules/inference/capability-transforms.rules.json:2-11`
> *"Gemma 4 E2B retained Q4K materialization produces NaN in L0.ffn_down on Apple Metal"*

The runtime's finiteness guard fires at prefill-sample under fail-fast policy. Even forcing `activationDtype: f32` doesn't help — the retain-q4k lane internally narrows to f16 somewhere, and the f16 op produces NaN on Metal but not RDNA3 (per `tests/config/capability-retain-q4k-policy.test.js`).

**Runtime knobs explored — no win**:
- batchSize 4/8/16/32 sweep: bs=8 already optimal (±5% noise)
- useOrtFlashPrefillAttention: −2.7% (attention pool too small)
- activationDtype=f16 selective decode lane: −8% (fires but doesn't help)
- TurboQuant KV profiles exist but KV is only 5 MB — not the bottleneck for short context

**Three real paths forward, in priority order:**

1. **Fix the Apple Metal NaN bug** (highest ROI, riskiest). Read `src/gpu/kernels/fused_matmul_q4_widetile.wgsl`, `_residual.wgsl`, `_f16.wgsl`, and `fused_ffn_q4k_f16.wgsl` line-by-line for f16 reads/writes/casts. Suspect Apple Metal-specific subnormal flush behavior or fastmath rounding. If a fix lands, drop the capability transform and unlock 30–50% throughput.

2. **f16a prefill + useFusedGateUpGelu** (medium scope). The `fused-gate-up-gelu.js` kernel exists, requires f16 weights + f16 activations + gelu (bypasses q4k path entirely). Currently the selective f16 lane only narrows decode activations. Extending to prefill + flipping `useFusedGateUpGelu: true` could be a 20–30% prefill win without touching the NaN bug.

3. **ReDrafter speculative decoding** (large scope, large win). Apple paper shows 2.3× on Apple Silicon via Metal/MLX. EAGLE-3 also reports 2–6×. Requires draft-head training/loading, verification path, KV rollback. Probably the biggest absolute win available but >>1-iter scope.

**Where things live:**
- Per-kernel profile JSON: `reports/perf/gemma-4-int4ple-loop/iter-002-profile.json`
- All sweep + verify outputs: `reports/perf/gemma-4-int4ple-loop/`
- Sweep harness: `reports/perf/gemma-4-int4ple-loop/iter-005-sweep.sh`
- This file (full per-iter detail below): `reports/perf/gemma-4-int4ple-loop/notes.md`

**State**: clean. No persistent edits to source. Capability rules file restored. No backup files left. The `.claude/scheduled_tasks.lock` is loop runtime state.

**Reproduce baseline**:
```
node src/cli/doppler-cli.js bench \
  --config '{"request":{"workload":"inference","modelId":"gemma-4-e2b-it-q4k-ehf16-af32-int4ple","runtimeProfile":"experiments/bench/gemma4-bench-q4k","cacheMode":"warm"},"run":{"surface":"auto","bench":{"save":true}}}' \
  --json
```

**Reproduce per-kernel profile**:
```
node src/cli/doppler-cli.js debug \
  --config '{"request":{"workload":"inference","modelId":"gemma-4-e2b-it-q4k-ehf16-af32-int4ple"},"run":{"surface":"auto"}}' \
  --runtime-config '{"shared":{"tooling":{"intent":"investigate"},"debug":{"profiler":{"enabled":true}}},"inference":{"prompt":"<long enough to fill prefill chunks>","sampling":{"temperature":0,"topK":1},"batching":{"maxTokens":8}}}' \
  --json
```

---


## Iter 1 — baseline + classification (2026-04-26)

### Headline numbers (most recent existing bench, 2026-04-24T22:32:40Z, profile `experiments/bench/gemma4-bench-q4k`)

| Metric                | Value          |
|-----------------------|---------------:|
| decode tok/s          | 10.60          |
| prefill tok/s         | 80.00          |
| prefill tok/s (TTFT)  | 79.53          |
| TTFT                  | 792 ms         |
| prefill ms            | 787.5 ms (63 tok) |
| decode ms             | 6036 ms (64 tok) |
| ms/decode-tok p50     | 94.3 ms        |
| model load            | 2133 ms        |
| peak pool             | 7.49 GB        |

User's smoke run (different prompt: 19 prefill tok): TTFT 879 ms, prefill 24.3 tok/s, decode 11.06 tok/s. Consistent with 80 tok/s steady-state once prefill amortizes.

### Manifest kernel routing (skill §2b high-ROI check) — looks correct

Execution graph at `models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/manifest.json`:

- **prefill / sliding-window layers (28 of 35)** → `attn_head256` = `attention_head256_f16kv.wgsl` ✓ fast variant for headDim=256
- **prefill / global layers (7 of 35: 4,9,14,19,24,29,34)** → `attn_head512` = `attention_head512_f16kv.wgsl` ✓ fast variant just shipped per Gemma 4 iter 25
- **prefill q/k/v/o projections** → `tiled` = `matmul_f16w_f32a.wgsl` (f16 weights, f32 activations) — implies q4 weights are materialized to f16 prior. Worth investigating whether fused `q4_widetile` direct-dequant is faster (kernels exist in repo but not registered in this manifest).
- **decode q/k/v/o projections** → `gemv` = `matmul_gemv_subgroup.wgsl` ✓ subgroup gemv
- **decode attention** → `attn_decode` = `attention_decode_online_f16kv.wgsl`
- **lm_head decode** → `lm_head_gemv_stable` = `matmul_gemv_subgroup.wgsl` (multicol, 64 cols/wg, 4 threads/col, f32 stability)
- **lm_head prefill** → `lm_head_prefill_stable` = `matmul_f16w_f32a.wgsl` (f32 stability)

No old-slow `attention_streaming_f16kv` or `fused_matmul_q4_batched_multicol_shared` rooted in the actual graph. (They are *registered* in the kernel map but not used by any step.) Manifest swap is **not** the iter 23–25 / Qwen-3.5 type win here.

### Decode wall classification — incomplete from existing bench

GPU breakdown from the bench:
- `decodeOrchestrationMs`: **6036 ms** (≈ 100% of decode time)
- `decodeRecordMs`: 0, `decodeSubmitWaitMs`: 0, `decodeReadbackWaitMs`: 0
- `gpu.prefillMs.samples`: 0, `gpu.decodeMs.samples`: 0 ← per-kernel GPU timings not captured in this run
- `singleTokenSubmitWaitMs`: 267.5 ms, `singleTokenReadbackWaitMs`: 3970 ms, `singleTokenOrchestrationMs`: 1797 ms (probe — separate from headline)

Conclusion: profiler not enabled in `experiments/bench/gemma4-bench-q4k` profile, so we cannot classify GPU compute vs CPU orchestration from this artifact. **Iter 2 must run a profiler-enabled probe.**

`decodeMode = "batched_gpu_stepwise_ple"` and `decodeBatchSize = 8` per execution contract. PLE prepared-token cache: **35 hits / 28 misses (44% miss rate)** over 63 tokens. PLE prep is a candidate CPU-side overhead.

### Existing assets discovered (do not re-invent)

Runtime profiles:
- `profiles/turboquant.json` — tiered KV layout, hotWindow=256, coldDtype=f16, turboquant compression bitWidth=4
- `profiles/turboquant-contiguous.json` — contiguous layout + turboquant kv quantization
- `profiles/turboquant-contiguous-prod.json` — same with prodMode=true (QJL residual)
- `profiles/gemma4-e2b-int4ple-f16a-probe.json` — narrows activation dtype to f16 (capability-aware kernel-path remap, model-pinned)
- `profiles/gemma4-e2b-prefill-profile.json` — GPU timestamp-query profiling enabled, 81-tok prompt, log.warn('Profile') receipt

WGSL kernels (already in repo, possibly unrouted for INT4 PLE):
- `attention_decode_contiguous_turboquant_f16kv.wgsl` / `_prod_f16kv.wgsl`
- `attention_decode_tiered_turboquant_f16kv.wgsl` / `_prod_f16kv.wgsl`
- `attention_decode_tiered_int4_f16kv.wgsl`, `_int8_f16kv.wgsl`
- `attention_prefill_flash_head256_f16kv.wgsl`, `attention_prefill_flash_reduce.wgsl` (flash prefill — also possibly unrouted)
- `fused_matmul_q4_widetile.wgsl`, `_widetile_f16.wgsl`, `_widetile_residual.wgsl` (widetile q4 — not in this model's graph)
- `fused_rmsnorm_q4_widetile.wgsl` (fused rmsnorm + q4 matmul)
- `fused_ffn_q4k.wgsl`, `fused_ffn_q4k_f16.wgsl`

Reports for this model: 19 bench JSONs from 2026-04-24 in `reports/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/`. Need to scan the variance to see which knobs were swept.

### Iter 1 conclusion

The "obvious" iter 23–25-style manifest swap doesn't apply — the model is already routed to the head256/head512 fast attention variants. The decode wall is not yet classified because the existing bench did not have profiler.enabled. Three concrete iter-2 priorities:

1. **Run a profiler-enabled probe** to capture per-kernel GPU walls for one decode step and one prefill chunk. Gate everything else on knowing whether the wall is GPU compute, submit, readback, or CPU-side PLE prep.
2. **Web research** the user's listed enhancements (TurboQuant, speculative decode, etc.) and 2025–2026 additions, score by (impact × WebGPU feasibility × Doppler integration cost).
3. **Run the existing TurboQuant profiles cold/warm against this model** and bench delta — they are runtime-config-only changes, zero code, zero risk to manifest.

### Open questions to track

- Why does `decodeOrchestrationMs` absorb 100% of decode time? Bug in metric attribution, or genuine CPU dominance? Profiler will tell.
- Q4 weights → f16 materialization happens where? (look at `src/inference/pipelines/text/init.js` weight-prep path or buffer-pool dequant)
- Is `fused_matmul_q4_widetile` faster than `matmul_f16w_f32a` for this model's prefill projection shapes (Q proj: 1536→2048; K/V proj: 1536→256; O proj: 2048→1536)?
- PLE 44% cache miss rate — is the cache too small for this prompt? 28 entries cached = 28 token-ids — for 63 prefill tokens with maybe repeated tokens.

---

## Iter 2 — profiler probe + web research (2026-04-26)

### Profiler-enabled run (investigate intent, 82-tok prompt, 32 decode)

Ran `doppler debug` with `runtime.shared.debug.profiler.enabled: true` against int4ple. Output saved to `reports/perf/gemma-4-int4ple-loop/iter-002-profile.json`. Note this is investigate mode, slower than the calibrate baseline by design.

Headline (investigate-mode, NOT comparable to calibrate baseline):
- decode 9.82 tok/s (vs 10.6 calibrate)
- prefill 68.05 tok/s (vs 80 calibrate)
- TTFT 1210 ms (vs 792 calibrate)

Wall classification (decode total 3257 ms, 32 tokens):
- GPU compute (`gpu.decodeMs`): 1970 ms = **60.5% of decode wall**
- CPU orchestration: ~1287 ms = **39.5% of decode wall**

Wall classification (prefill total 1205 ms, 82 tokens):
- GPU compute (`gpu.prefillMs`): 1028 ms = 85% of prefill wall
- `prefillSubmitWaitMs`: 926 ms (much of this overlaps with GPU compute — dispatch queueing pattern)
- `prefillRecordMs`: 57 ms

### Per-kernel hot ops — the real wall

**Prefill GPU breakdown (1028 ms over 35 layers / 9 chunks):**

| Op                                | total ms | count | avg ms | % of GPU |
|-----------------------------------|---------:|------:|-------:|---------:|
| matmul:ffn_down                   |   257.03 |    35 |  7.344 |   25.0%  |
| matmul:ffn_gate                   |   251.99 |    35 |  7.200 |   24.5%  |
| matmul:ffn_up                     |   247.86 |    35 |  7.082 |   24.1%  |
| **FFN total**                     | **757**  |  105  |        | **73.6%**|
| matmul:q_proj                     |    66.06 |    35 |  1.887 |    6.4%  |
| matmul:o_proj                     |    63.64 |    35 |  1.818 |    6.2%  |
| attention                         |    46.07 |     9 |  5.119 |    4.5%  |
| matmul:per_layer_model_projection |    32.18 |    35 |  0.919 |    3.1%  |
| rmsnorm                           |    13.04 |     9 |  1.449 |    1.3%  |
| cast_f32_to_f16                   |    12.65 |     9 |  1.405 |    1.2%  |
| scale, gelu, residual, rope, gather, k_proj, v_proj | <11   |       |       |  rest    |

K/V projections are tiny (4 ms each) because of MQA (1 KV head, headDim=256).

**Decode GPU breakdown (1970 ms over 32 decode tokens × 35 layers):**

| Op                                | total ms | count | avg ms | % of GPU |
|-----------------------------------|---------:|------:|-------:|---------:|
| matmul:ffn_down                   |   383.78 |  1085 |  0.354 |   19.5%  |
| matmul:ffn_gate                   |   372.05 |  1085 |  0.343 |   18.9%  |
| matmul:ffn_up                     |   371.65 |  1085 |  0.343 |   18.9%  |
| **FFN total**                     | **1128** |  3255 |        | **57.2%**|
| matmul:lm_head                    |   282.26 |    31 |  9.105 |   14.3%  |
| matmul:o_proj                     |   102.69 |  1085 |  0.095 |    5.2%  |
| matmul:q_proj                     |    99.55 |  1085 |  0.092 |    5.1%  |
| matmul:per_layer_input_gate       |    98.24 |  1085 |  0.091 |    5.0%  |
| attention                         |    79.36 |    31 |  2.560 |    4.0%  |
| matmul:per_layer_projection       |    44.70 |  1085 |  0.041 |    2.3%  |
| rmsnorm                           |    44.24 |    31 |  1.427 |    2.2%  |

Note `count=1085` = 31 tokens × 35 layers — confirms one dispatch per layer per decode step.

### KernelSelect logs (info-level) — what is *actually* dispatched

From info-level run:
```
matmul variant=f32 reason=default          (initial registration default)
matmul variant=f16w_f32a reason=path_override   ← FFN + Q/O proj prefill
matmul variant=gemv_subgroup_multicol reason=path_override   ← lm_head
matmul variant=gemv_subgroup_vec4 reason=path_override        ← Q/O proj decode
matmul variant=gemv_subgroup reason=gemv                       ← K/V proj decode
attention variant=prefill_head256_f16kv reason=path_override:tiled_small
attention variant=prefill_head512_f16kv reason=path_override:tiled_small
attention variant=decode_online_f16kv reason=path_override:subgroup
```

**Key conclusion:** all FFN matmuls in prefill go through `matmul_f16w_f32a.wgsl` — meaning Q4_K weights are dequantized to f16 (somewhere on the load/init path) and then matmul'd. The `fused_matmul_q4_widetile.wgsl`, `fused_matmul_q4_widetile_f16.wgsl`, `fused_matmul_q4_widetile_residual.wgsl`, `fused_rmsnorm_q4_widetile.wgsl`, `fused_ffn_q4k.wgsl`, `fused_ffn_q4k_f16.wgsl` kernels exist but are dormant for this model — no rule routes to them and `src/rules/**` does not mention them.

This is a different shape from the iter 23–25 / Qwen-3.5 fix (those swapped manifest entries that referenced specific kernel names). Here the manifest's `ffn` step uses `gelu` as the registered kernel; the runtime expands it into separate ffn_gate/ffn_up/ffn_down matmul calls and routes those via the matmul rule registry → `f16w_f32a`. So the swap will be at the JS expansion site, not just a manifest digest swap.

FFN expansion sites: `src/inference/pipelines/text/ffn/dense.js` and `src/inference/pipelines/text/ffn/standard.js`.

### Web research — 2026 candidate enhancements (scored)

Scoring axes (each 1–5, higher better):
- **Impact**: expected speedup against the identified bottleneck
- **WebGPU fit**: how cleanly it maps to WebGPU/Metal capabilities (subgroups, f16, no CUDA-only ops)
- **Doppler integration cost**: lower is better (existing kernels = high; new WGSL = medium; new pipeline phase = low)

| Candidate                                          | Impact | WebGPU fit | Integration | Notes |
|----------------------------------------------------|:------:|:----------:|:-----------:|-------|
| **Wire fused_ffn_q4k_f16 into FFN expansion**      |   5    |     5      |      4      | Kernel exists; replaces 3 matmul + activation dispatches with 1; targets the 57–74% of GPU. NO new WGSL. |
| **Wire fused_matmul_q4_widetile for q/o/lm_head**  |   3    |     5      |      4      | Skips f16 dequant materialization for q_proj/o_proj/lm_head; smaller targets. |
| **Wire fused_rmsnorm_q4_widetile**                 |   2    |     5      |      3      | RMSNorm is only 1.3% prefill / 2.2% decode; saves dispatches but kernel-time gain modest. |
| **TurboQuant KV cache (already in repo profiles)** |   2    |     4      |      5      | Runtime-config-only swap (`profiles/turboquant-contiguous-prod`); KV is small here (5 MB) so memory not the wall; could help at long context. |
| **Speculative decoding (ReDrafter / EAGLE-3)**     |   4    |     3      |      1      | 2–3× decode wins on Apple Silicon proven; needs draft head + verification path; large scope. |
| **Mega-kernel / WeInfer-style persistent dispatch**|   4    |     2      |      1      | Targets dispatch overhead (40% CPU side); needs WebGPU runtime rework, browser/spec dependencies. |
| **Flash prefill (`attention_prefill_flash_head256`)** | 2   |     5      |      4      | Kernel exists but attention is only 4.5% of prefill — small win. |
| **Move PLE prep off the critical path**            |   2    |     4      |      3      | 75% miss rate on decode is real, but the GPU walltime calls show PLE matmul ops are <8% of decode — bigger fish first. |

### Iter 2 conclusion → iter 3 target

**#1 candidate: Wire `fused_ffn_q4k_f16.wgsl` into the FFN expansion path** for this model.

Rationale:
- Targets 57.2% of decode GPU and 73.6% of prefill GPU — by far the biggest pool.
- Three dispatches per layer collapse to one (or one + activation) — direct dispatch-overhead win, which the WebLLM/WeInfer literature confirms is the WebGPU dominant cost.
- Kernel already exists in `src/gpu/kernels/`. No new WGSL. The work is JS plumbing in `src/inference/pipelines/text/ffn/`.
- f16 weight variant (`fused_ffn_q4k_f16.wgsl`) means we keep the f16 activation path the existing `f16w_f32a` matmul uses, minimizing precision drift.
- Correctness gate: `doppler verify` against the existing reference transcript. If MATCH, run bench delta.

Iter 3 work:
1. Read `src/inference/pipelines/text/ffn/dense.js` + `standard.js` to find the matmul-call site.
2. Read the matmul rule registry to confirm there's no easier route.
3. Read `fused_ffn_q4k_f16.wgsl` to confirm interface (inputs: input, gate_w, up_w, down_w; outputs).
4. Decide approach: rule-map override vs JS expansion change vs manifest kernel-ref addition.
5. Make minimal change, run verify, run bench, record delta.

### Sources (web research)

- [TurboQuant — Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — randomized Hadamard rotation + Lloyd-Max quantizer, 6× KV memory reduction, 8× attention speedup on H100, ICLR 2026.
- [TurboQuant llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/20969) — community implementations.
- [TurboQuant InfoQ summary](https://www.infoq.com/news/2026/04/turboquant-compression-kv-cache/)
- [EAGLE-3 (NeurIPS'25)](https://github.com/SafeAILab/EAGLE) — speculative decoding, 2–6× faster.
- [Apple ReDrafter](https://machinelearning.apple.com/research/recurrent-drafter) — 2.3× speedup on Apple Silicon via Metal/MLX.
- [WebGPU dispatch overhead paper (arxiv 2604.02344)](https://arxiv.org/abs/2604.02344) — per-op overhead dominates; structural fusion (RMSNorm, MLP) is the lever; Metal backend often shows no benefit from elementary fusion.
- [WebLLM (arxiv 2412.15803)](https://arxiv.org/abs/2412.15803) — WebGPU LLM inference engine baseline.
- [Megakernel Mirage compilation](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17) — long-term direction; not iter-3 territory.

---

## Iter 3 — wire fused FFN; hit pre-existing Apple Metal NaN guardrail (2026-04-26)

### What I tried

1. Read FFN expansion sites (`src/inference/pipelines/text/ffn/{standard,dense}.js`) and the candidate kernel (`src/gpu/kernels/fused_ffn_q4k_f16.wgsl`).
2. Found the activation gate for the fused-q4k FFN path:
   - `useFusedGateUp` rule (`src/rules/inference/ffn.rules.json`) requires `hasQ4KMaterialization=true` + `activationDtype=f32` + `hiddenSizeAligned32` (third match clause).
   - `hasQ4KMaterialization` is set by `resolveFusedGateUpWeights()` in `dense.js:238` — requires `weight.materializations?.q4k?.buffer` for both gate and up.
   - This is gated by `retainQ4KMaterialization` session flag (default false).
   - The model's conversion config has `useWideTileQ4KPrefill: true` AND `retainQ4KMaterialization: false` (mismatch) — meaning widetile is also dead unless retained.
3. Ran `doppler verify` with `--runtime-config '{"inference":{"session":{"retainQ4KMaterialization":true}}}'`. Verify PASSED but numbers ≈ baseline. KernelSelect logs revealed why.

### Pre-existing Apple Metal NaN guardrail

`src/rules/inference/capability-transforms.rules.json:2-11`:
```json
{
  "match": {
    "modelId": { "startsWith": "gemma-4-e2b-it-q4k-" },
    "platformVendor": "apple",
    "retainQ4KMaterialization": true
  },
  "transforms": ["disableRetainQ4KMaterialization"],
  "reason": "Gemma 4 E2B retained Q4K materialization produces NaN in L0.ffn_down on Apple Metal"
}
```

Test fixture confirms the rule scope: AMD RDNA3 hits no transform, Apple metal-3 hits `disableRetainQ4KMaterialization` (`tests/config/capability-retain-q4k-policy.test.js`).

This single guardrail is suppressing an **entire family of optimizations** on Apple Metal for Gemma 4 E2B Q4K models:
- `fused_ffn_q4k_f16.wgsl` — the 57.2% decode / 73.6% prefill GPU pool target.
- `fused_matmul_q4_widetile.wgsl`, `_widetile_f16.wgsl`, `_widetile_residual.wgsl` — q/o proj + ffn matmul widetile.
- `fused_rmsnorm_q4_widetile.wgsl` — fused rmsnorm + widetile.
- All of which require q4k buffers retained.

### Tried fallback — selective f16 decode lane

`useGemma4Int4PleSelectiveF16Decode` capability transform fires when `activationDtype=f16` is requested. KernelSelect log diff:
- decode matmul: `gemv_subgroup_vec4` → `gemv_subgroup_vec4_f16a`
- decode attention: `decode_online_f16kv` → `decode_online_f16` (full f16)
- prefill: unchanged (still `f16w_f32a`)

Verify PASS, but decode tok/s 9.65 vs 10.07 baseline — within noise, not a perf win.

### Iter 3 result table

| Trial                                                | decode tok/s | prefill tok/s | TTFT  | passed | notes |
|------------------------------------------------------|-------------:|--------------:|------:|:------:|-------|
| baseline (no override)                               |        10.07 |         42.11 |  543  | yes    | f16w_f32a matmul, fused FFN OFF |
| retainQ4KMaterialization=true                        |        10.07 |         42.11 |  543  | yes    | guardrail nullifies override; no actual change |
| activationDtype=f16 (selective f16 decode lane)      |         9.65 |         42.25 |  540  | yes    | f16a kernels fire on decode; no measurable win |

(All trials use 19-token prompt "The color of the sky is", 32 decode tokens. Numbers are calibrate-mode comparable.)

### Iter 3 conclusion → iter 4 target

The single highest-ROI fix is **investigating and resolving the L0.ffn_down NaN on Apple Metal**. This unlocks the entire fused-q4k optimization stack:
- Estimated win: 30–50% throughput improvement if the FFN fusion + widetile + rmsnorm fusion all become live (combined targets >80% of the GPU pool).
- Risk: it's a real correctness bug, not a config issue. The guardrail says NaN appears at L0.ffn_down — likely a precision overflow or subnormal handling in the kernel, possibly Metal-specific.

Iter 4 plan (per CLAUDE.md correctness discipline):
1. Hypothesis first: NaN in L0.ffn_down is most likely caused by (a) f16 overflow in the Q4_K dequantize+accumulate path with f32 activations, (b) workgroup storage race when reusing scratch buffers across gate/up/down dispatches, or (c) Metal-specific subnormal flush behavior.
2. Run `doppler-debug` per-layer probe with retainQ4KMaterialization forced true (override the capability transform) and capture L0.ffn_down output. Compare against reference (CPU torch baseline or no-retention path output).
3. Scope: `src/gpu/kernels/fused_ffn_q4k_f16.wgsl` (lines covering ffn_down — not in this kernel; ffn_down has its own widetile kernel).
4. If fix lands, run verify on Apple Metal, run bench delta, drop the capability transform.

If the bug is too deep for iter 4, fallback candidates (in order):
- A. `useFusedGateUpGelu: true` — different fused kernel (`fused-gate-up-gelu.js`) for f16 weights + f16 activations + gelu (prefill only). Doesn't depend on q4k retention. But requires shifting prefill activations to f16 — additional work and the existing selective-f16 lane is decode-only.
- B. `useWideTileResidualFusion` + `useFusedRmsnormWideTile` — also blocked by retainQ4KMaterialization; same Apple Metal guardrail kills these.
- C. lm_head decode kernel tuning (14.3% of decode; standalone GEMV op).
- D. `useOrtFlashPrefillAttention` — only 4.5% of prefill; smaller pool but no q4k dependency.

Files to read in iter 4: `src/config/transforms/execution-graph-transforms.js:1865` (the `disableRetainQ4KMaterialization` transform body), the relevant `fused_*_q4_*.wgsl` kernels, and `src/inference/pipelines/text/init.js:283` (where the transform is consumed).

---

## Iter 4 — NaN guardrail confirmed real; kernel-level cause not yet localized (2026-04-26)

### Hypothesis (one, falsifiable)

NaN in L0.ffn_down originates from f16 accumulator overflow in `fused_matmul_q4_widetile*.wgsl` when fed gelu-activated values, OR from the fused-q4k FFN gate+up step producing Inf/NaN that propagates. Diagnostic: probe L0 ffn_in / ffn_gate / ffn_up / ffn_act / ffn_out boundaries with retainQ4K forced on. Falsified if NaN appears at ffn_in (then it's an upstream attention bug, not an FFN bug).

### Setup

To bypass the `disableRetainQ4KMaterialization` capability transform during investigation, I temporarily edited `src/rules/inference/capability-transforms.rules.json` to disable the matching rule (changed modelId `startsWith` to a sentinel that won't match). The file was reverted after each probe run; backup file removed. **No persistent edit landed.**

### What happened — the finiteness guard fires before probes can read

With retainQ4K=true and rule disabled, the run hits:
```
[Pipeline] prefill-sample: finiteness guard triggered for kernelPath
"gemma-4-e2b-it-q4k-ehf16-af32-int4ple-execution-inline" under fail-fast policy.
Resolve the unstable path with an explicit capability-aware execution override,
or opt into alternate-plan recovery with
runtime.inference.compute.rangeAwareSelectiveWidening.onTrigger="fallback-plan".
```

Tried the recovery path:
```json
{ "compute": { "rangeAwareSelectiveWidening": {
    "enabled": true, "includeNonFinite": true,
    "onTrigger": "fallback-plan", "absThreshold": 65504
}}}
```

Same finiteness-guard error fired. Either the fallback plan is failing to construct, or `onTrigger=fallback-plan` only changes the post-trigger handling but the guard itself still throws if no recovery succeeds.

### What we now know for sure

1. The NaN guardrail (`disableRetainQ4KMaterialization` capability transform) is protecting against a **real** correctness failure — the runtime's finiteness guard catches it at `prefill-sample`. This is not over-conservatism.
2. NaN propagates through prefill into the LM head, surfacing at sampling. So either:
   - NaN is born somewhere before lm_head (most likely L0 layers per the rule's reason text)
   - Lm_head itself produces non-finite logits when fed slightly-anomalous-but-finite hidden states
3. `finitenessGuardEnabled = activationDtype === 'f16' && finitenessPolicy.enabled` (`execution-plan.js:124`). The guard fires only on f16 activations. Since our config is f32, **the f32 path internally widens to f16 somewhere on the retain-q4k lane**, then fails the guard. This means: the bug is f16-precision-specific. With f32 activations all the way, retain-q4k might actually work — but I haven't been able to isolate that lane in this iter.
4. The whole optimization stack (fused FFN q4k, widetile q4 prefill, widetile residual fusion, fused rmsnorm widetile) all depend on retainQ4KMaterialization=true. Apple Metal cuts all of them. AMD RDNA3 keeps them (per `tests/config/capability-retain-q4k-policy.test.js`).

### Implication for the perf loop

Fixing this NaN unlocks 30–50% throughput on Apple Metal — high ROI. But the root cause is in the kernel/dtype interaction, not config. Iter 5 needs either a pure-f32 retain-q4k probe (test #3 above) OR an actual kernel-level fix.

### Iter 5 candidates

A. **Try retainQ4K + force f32 throughout** — `runtime.inference.compute.activationDtype: f32` (already default) + force any remaining f16 narrowing off via kernel-path overrides, then run probe. If this works without NaN and is faster than baseline, that's the win without kernel work.

B. **Pivot to non-q4k targets** — runtime-only knobs that don't depend on retainQ4KMaterialization:
   - PLE cache sizing (75% decode miss rate currently)
   - decode batchSize tuning (`session.decodeLoop.batchSize`)
   - lm_head GEMV constants (14.3% of decode time)
   - Flash prefill attention `useOrtFlashPrefillAttention` (only 4.5% of prefill, small target)
   - useFusedGateUpGelu — needs f16 weights, needs prefill activations f16; would require pipeline narrowing, large change

C. **Investigate the kernel directly** — read `fused_matmul_q4_widetile.wgsl` line-by-line + the kernel's accumulator path, look for f16/f32 conversions. Match against Apple Metal's documented quirks (subnormal flush, fast-math). High-cost, high-leverage if successful.

For iter 5, going with **A first (lowest cost, highest information)**, then B (PLE cache + lm_head — quick runtime knobs), reserving C for later.

---

## Iter 5 — pure-f32 retain-q4k still NaN; runtime sweep flat (2026-04-26)

### Plan A — pure-f32 retain-q4k probe

Disabled the capability transform (rule-edit, reverted after run; backup file removed). Ran with explicit `runtime.inference.compute.activationDtype: "f32"` + `retainQ4KMaterialization: true`. Same finiteness-guard error fires at prefill-sample.

This proves: even when the *requested* activationDtype is f32, the retain-q4k execution lane internally narrows activations to f16 at one or more matmul boundaries. The NaN is born inside the lane regardless of caller's intent. Only a kernel/dtype-policy fix can resolve it.

### Plan B — runtime knob sweep (no q4k dep)

Tight, cache-warm bench sweep on Node surface, deterministic generation (seed/temp=0/topK=1), 64 decode tokens:

| trial             | decode tok/s | prefill tok/s | TTFT (ms) | decodeMs | vs baseline |
|-------------------|-------------:|--------------:|----------:|---------:|------------:|
| baseline (bs=8)   |        10.53 |          56.6 |       354 |    12160 |          —  |
| bs=4              |        10.19 |          55.6 |       363 |    12567 |       −3.2% |
| bs=16             |         9.92 |          56.1 |       357 |    12902 |       −5.8% |
| bs=32             |        10.06 |          56.1 |       358 |    12720 |       −4.5% |
| useOrtFlashPrefillAttention | 10.25 |        56.4 |       356 |    12491 |       −2.7% |

(prefill rate is lower than the 80 tok/s in the older 63-tok-prompt benches because this sweep used a 7-tok prompt where startup overhead dominates the rate calc.)

**Headline**: bs=8 is already the best decode-cadence choice; flash prefill doesn't move the needle (attention is only 4.5% of prefill GPU); larger batch sizes hurt slightly (ringbuffer pressure or readback-coalesce loss). All deltas are within sample noise (~±5%).

### Iter 5 conclusion

Decode wall on this model + this hardware is genuinely GPU-compute-bound (FFN + lm_head, ~70%+). Orchestration is already tight. Runtime-only knobs that don't unlock the q4k-fusion path can't materially move throughput.

Three real paths forward, all expensive:

1. **Fix the Apple Metal NaN** — kernel deep-dive on `fused_matmul_q4_widetile*.wgsl` and `fused_ffn_q4k_f16.wgsl`. Find the exact f16 op that breaks on Metal. This unlocks 30–50% throughput per the iter-2 GPU breakdown.
2. **Implement ReDrafter / EAGLE-3 speculative decoding** — proven 2–2.3× win on Apple Silicon. Substantial architectural work (draft head training/loading, verification path, cache rollback). Probably the largest absolute win available, but >> one-iter scope.
3. **Reduce prefill activation precision (f16 widening) via the existing selective lane** — the `gemma4-e2b-int4ple-f16a-probe` path with prefill activations narrowed to f16 might unlock `useFusedGateUpGelu` (which needs f16 weights + f16 activations + gelu, no q4k dep). Requires the f16-narrowing transform on prefill, plus correctness gating. Worth one trial.

**Iter 6 target**: option 1 — read the widetile q4 kernel(s) for the specific f16 failure mode. Identify the line. If it's a small fix (e.g., promote an f16 accumulator to f32, or add a guard against subnormal flush), propose it. If it's structural, document the work-required estimate and pivot to option 3 (f16a + useFusedGateUpGelu) for iter 7.

### Cumulative delta table (iters 1–5)

| iter | trial / target                                         | decode tok/s | passed? |
|------|---------------------------------------------------------|-------------:|:-------:|
| 1    | baseline (existing 2026-04-24 bench)                   |        10.60 |   yes   |
| 3    | retainQ4KMaterialization=true (override)               |        10.07 | yes (guardrail nullifies) |
| 3    | activationDtype=f16 (selective decode lane)            |         9.65 |   yes   |
| 4    | retainQ4K=true with rule disabled (probe)              |       —      | NO (finiteness guard) |
| 5    | retainQ4K=true + activationDtype=f32 + rule disabled   |       —      | NO (finiteness guard) |
| 5    | bs=4 / bs=16 / bs=32 / ortFlashPrefill                  |   9.92–10.25 |   yes   |

No win shipped yet. The wall is real and clean: ~70% of GPU compute is on q4k-fusion-blocked kernels.

