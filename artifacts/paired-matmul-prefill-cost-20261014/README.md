# Where paired matmul prefill time goes (2026-10-14)

**Question:** run 7 of the installed journey measured `intel-host` prefill at
1.0-1.1 s per prompt token, the largest stage cost on placement B. The
qualified Capsule (`chunked-prefill-capsule-20261014`) runs its projections
through `src/gpu/kernels/matmul_paired.wgsl` (sha256
`2d56380990eb5c66a8272ca8f9ff3903bd55f018f6d9de079cc32ae5759aafb7`, the same
bytes as the final-9 install). Is that kernel bound by its compensated two-F32
arithmetic (`pair_add`, `pair_mul_scalar` with integer-based
`addition_residual` and `product_residual`), or by the Q4_K weight reads and
decoding? The answer decides whether weight reuse across prefill tokens could
help.

**Method:** [paired-matmul-bench-node.js](paired-matmul-bench-node.js) times
one dispatch of the shipped kernel against two controls built from the same
text:

- `plain-f32`: the same input reads and Q4_K weight decoding, with a plain
  F32 multiply-add.
- `paired-no-weight`: the same compensated arithmetic, with no weight read or
  decoding.

The weights are random Q4_K blocks (`WEIGHT_Q4K`, `TRANSPOSE_B`, 64-lane
workgroups, one workgroup per output). The token counts are M = 1, 4 and 16
(16 is the Capsule's prefill chunk). Each run reports the median of 5 timed
submissions after 2 warm-ups.

- **Runtime:** Node v22.21.0 with the `webgpu` 0.4.0 Dawn module from the
  final-9 install, the same stack as the native stages.
- **Bounds:** a bounded systemd scope on each host (10 GiB high, 12 GiB max,
  no swap, 240 s).
- **GPU checks:** no validation errors in any run. The i915 hang/reset count
  on `intel-host` stayed at 54 and the amdgpu count on `amd-host` stayed at
  50.
- **Not isolated:** the public contributors on both hosts kept running.

This is diagnostic timing, not numerical qualification.

## Results (median ms per dispatch)

| Host, GPU | Shape (N x K) | Variant | M=1 | M=4 | M=16 |
|---|---|---|---|---|---|
| `intel-host`, Intel Gen12LP (Tiger Lake GT2), Mesa 24.2.8 | 3584 x 1024 | shipped | 19.43 | 75.03 | 301.32 |
| | | plain-f32 | 1.08 | 3.59 | 14.18 |
| | | paired-no-weight | 18.05 | 70.91 | 277.25 |
| | 1024 x 3584 | shipped | 17.18 | 63.91 | 275.34 |
| | | plain-f32 | 0.81 | 3.01 | 11.78 |
| | | paired-no-weight | 15.96 | 62.18 | 243.90 |
| `amd-host`, AMD RDNA3, RADV Mesa 26.0.3 | 3584 x 1024 | shipped | 1.02 | 3.81 | 14.90 |
| | | plain-f32 | 0.09 | 0.17 | 0.52 |
| | | paired-no-weight | 0.91 | 3.43 | 13.64 |
| | 1024 x 3584 | shipped | 1.00 | 3.75 | 14.79 |
| | | plain-f32 | 0.11 | 0.19 | 0.58 |
| | | paired-no-weight | 0.88 | 3.87 | 13.75 |

The raw results are in `intel-host/` and `amd-host/`.

## Findings

1. **The kernel is bound by its compensated arithmetic.** Without any weight
   read, the arithmetic alone takes 88-93% of the shipped time on both GPUs.
   The same reads and decoding with plain F32 math take 4-6% on Intel. On AMD
   they take 4% at M=16; the M=1 runs there last about 0.1 ms, near the timer
   resolution.
2. **Time scales linearly with tokens** (about 18.8 ms per token per 3584 x
   1024 matrix on Intel), because each output's arithmetic is repeated per
   token. Weight reuse across prefill tokens would leave about 92% of the
   time in place, so it is not the fix.
3. **Intel Gen12LP runs this kernel about 20x slower than AMD RDNA3**
   (301 ms against 14.9 ms at M=16).
4. **The three feed-forward projections alone (gate, up, down)** cost about
   55 ms per prompt token per layer on Intel. Over stage 1's eight layers that
   is about 0.44 s of the measured 1.0-1.1 s per prompt token. The attention
   and linear-attention projections run through the same kernel and were not
   timed separately.

## What this leaves open

Two directions remain. Each one changes a qualified configuration and needs
its own qualification:

- **Placement.** Give the Intel stage fewer layers, or none, so that prepared
  routes rank by measured complete-route cost (Launch 3). This needs
  qualification of the new partition and no change to the numerics.
- **Arithmetic.** Make the compensated accumulation cheaper while staying
  inside the paired tolerance, for example by compensating once per Q4_K
  sub-block instead of once per multiply-add. This changes the numerics, so
  it needs numerical requalification against the frozen reference.
