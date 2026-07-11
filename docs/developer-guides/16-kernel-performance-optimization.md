# Kernel Performance Optimization

## Goal

Optimize an already-correct GPU execution path while preserving deterministic
output, explicit kernel selection, and claim-grade benchmark evidence.

## When To Use This Guide

Use this guide when a model passes deterministic verification but remains behind
a reference engine on a specific hardware and workload lane.

Do not use it to route around a stale artifact, a manifest mismatch, incoherent
output, or an invalid comparison contract. Follow
[Model Failure Action Plan](../model-failure-action-plan.md) until correctness
and artifact identity are established first.

## Blast Radius

- WGSL + kernel wrapper + runtime profile + registry + GPU tests + benchmark evidence

## Required Touch Points

- Kernels under `src/gpu/kernels/`
- Wrapper and dispatch selection under `src/gpu/kernels/*.js`
- Runtime profiles under `src/config/runtime/profiles/`
- `src/config/kernels/registry.json` and pinned kernel digests
- CPU-reference GPU tests under `tests/inference/` or `tests/gpu/`
- Vendor comparison config and saved receipts under `benchmarks/vendors/`

## Baseline Contract

Before changing a kernel:

1. Lock the model artifact, prompt, sampling tuple, token budgets, load mode,
   decode cadence, browser, and hardware.
2. Save a deterministic greedy run with exact token IDs for at least 128 decode
   tokens.
3. Capture the resolved runtime profile and active kernel variants. A requested
   variant that silently falls back is not optimization evidence.
4. Record phase timing and dispatch count per token before attributing the gap
   to arithmetic.
5. Change one lever at a time and retain the baseline result beside the candidate.

## Recommended Order

1. Decompose the token into projections, FFN, attention, LM head, sampling, and
   encoding/submission cost.
2. Tune the dominant Q4_K projection or fused-FFN shape only when that phase is
   measurably behind.
3. Preserve f32 accumulation and prove numeric parity before changing storage or
   activation dtype.
4. Count workgroups, barriers, dispatches, and submissions before accepting a
   locally faster inner loop.
5. Treat fused LM-head argmax as a greedy-only specialization with its own
   correctness gates.
6. Profile RoPE, softmax, and KV-cache reads when projection math no longer
   explains the residual.
7. Close the comparison with the paired statistical protocol in
   [Benchmark Methodology](../benchmark-methodology.md).

## Optimization Levers

### Q4_K Decomposition

For tall, single-token Q4_K projections, lane-to-row decomposition matters more
than nominal workgroup size. The Apple Metal Qwen 3.5 2B result used 16 subgroup
lanes per output row for the decode projections and fused gate/up path. Each lane
dequantizes and accumulates its share in f32, then the row reduces in f32.

Keep these properties explicit:

- Q4_K block layout and scale/min decoding match the established reference path.
- A full-block specialization is enabled only when the hidden dimension is an
  exact multiple of the Q4_K block size.
- Partial blocks use bounds-safe activation loads.
- Workgroup constants live in the runtime profile and are recorded in evidence.
- Unsupported subgroup capability or an unresolved named variant fails closed.

### Argmax Reduction

A fused LM-head argmax is valid only for deterministic greedy decode. Preserve
the materialized-logits path for temperature, top-k, top-p, or downstream logit
consumers.

Both reduction phases must use the same `(value, index)` operator:

- the higher value wins
- an exact tie selects the lower vocabulary index
- the identity is `(-inf, UINT_MAX)`
- absent vocabulary rows are padded with `-inf`, never zero

For a 16-lane row reduction, shuffle offsets must remain `1`, `2`, `4`, and `8`.
An offset of `16` merges adjacent rows on a 32-wide Apple subgroup. Do not assume
a subgroup width: gate the variant on the observed width, record it in the
profile/evidence, and fail closed when the layout contract is not met.

The reduction can still lose overall even when its dot product is efficient.
Count the phase-one workgroups and the second dispatch before promoting it.

### Accumulation Dtype

Dequantize, multiply, and reduce Q4_K partials in f32. Do not downcast lane
partials before the tree reduction. A different f32 summation order can create a
small rounding delta, so CPU-reference tolerances and long token-ID parity remain
required. F16 accumulation across a full hidden dimension is not an acceptable
shortcut for an exact greedy lane.

### Launch And Fusion

Kernel time is not the only decode cost. For each candidate record:

- dispatches per token
- workgroups per dispatch
- command-buffer submissions per token
- barriers introduced by the decomposition
- phase timing before and after the change

A decomposition that changes roughly 2,000 workgroups into roughly 16,000 can
lose despite doing less work per row. Once projection math is competitive,
compare encoded dispatch counts with the reference engine. A lower reference
count points to operation fusion, not another Q4_K inner-loop rewrite.

### Attention

Projection tuning does not cover RoPE, softmax, KV-cache reads, or attention
output assembly. Profile attention separately. If its decode share is larger
than the reference engine's, inspect contiguous versus strided KV access, cache
layout, read width, and dispatch fusion before revisiting projection kernels.

## What We Ruled Out

These results came from the Qwen 3.5 2B Apple Metal investigation. They are
negative evidence for that shape, not universal bans.

| Candidate | Observed result | Reuse rule |
| --- | --- | --- |
| F16 projection/activation path | Regressed or failed to improve the exact-parity decode lane | Do not trade accumulation precision for a nominally cheaper dtype without phase evidence and token parity |
| Dense F16 down-projection weights | Required about 6.40 GB and did not produce a usable end-to-end win | Reject materialization that removes the memory advantage of the Q4_K candidate without a measured decode gain |
| Workgroup geometry sweeps | 64 and 128 threads were neutral-to-worse; 256 remained best for the promoted SIMD16 path | Sweep geometry after fixing decomposition, and retain only a repeatable end-to-end gain |
| Shared-memory and barrier rearrangements | Did not explain the remaining decode delta | Treat barrier removal as a measured hypothesis, not an assumed win |
| SIMD16 fused LM-head argmax | Increased phase-one workgroups from about 2,000 to about 16,000 and reduced decode from 38.83 to 38.17 tok/s | Reject an inner-loop win when grid expansion loses end to end |
| Paired-row LM-head variant | Produced 38.75 versus the 38.83 tok/s baseline | Barrier/layout changes inside LM head were within noise and not promotable |
| Smaller submission cadence | Decode already used one submission, so submission count did not explain the residual | Inspect dispatch count and untouched attention work next |

Do not hand-copy these numbers into the generated competition scoreboard. Saved
benchmark receipts remain the source for generated model/platform results.

## Verification

Minimum kernel gates:

- deterministic CPU-reference comparison for a full-block Q4_K shape
- deterministic CPU-reference comparison for a partial-tail shape
- comparison with the established GPU kernel where one exists
- exact 128-token greedy output parity after profile wiring
- an interleaved long-form paired comparison before a parity or win claim

Current SIMD16 regression anchors:

- `tests/inference/fused-matmul-q4-metal-simd16-gpu-regression.test.js`
- `tests/inference/fused-ffn-q4-metal-simd16-gpu-regression.test.js`

When adding a subgroup fused-argmax variant, also add:

- an exact tie split across phase-one workgroups; the lower index must win
- a vocabulary length of `16 * N + 3`; padded lanes must never win
- a subgroup-width rejection test
- a 128-token exact golden-parity run

Run the focused GPU and profile tests, then the kernel registry/digest checks and
the repository green gate. GPU-required tests must skip explicitly on unsupported
hosts rather than becoming CI requirements.

## Common Misses

- Measuring a shader in isolation while ignoring grid size and launch count.
- Changing multiple projections, cadence, and prompt shape in one comparison.
- Assuming a profile key loaded without inspecting the resolved runtime report.
- Using zero as an argmax tail value or reduction identity.
- Letting a greedy-only fused path run for probabilistic sampling.
- Reporting a point estimate when the paired confidence interval crosses zero.
- Re-running a disproved projection variant after the residual moved to launch or
  attention cost.

## Related Guides

- [01-runtime-profile.md](01-runtime-profile.md)
- [06-kernel-path-config.md](06-kernel-path-config.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [13-attention-variant.md](13-attention-variant.md)
- [14-quantization-format.md](14-quantization-format.md)
- [Model Onboarding Playbook](model-onboarding-playbook.md)
- [Local GPU Challenger Framework](../local-gpu-challenger-framework.md)

## Canonical References

- [Kernel Testing Design](../kernel-testing-design.md)
- [Benchmark Methodology](../benchmark-methodology.md)
- [WGSL Style Guide](../style/wgsl-style-guide.md)
- `src/config/kernels/registry.json`
- `src/config/kernels/kernel-ref-digests.js`
- `benchmarks/vendors/README.md`
