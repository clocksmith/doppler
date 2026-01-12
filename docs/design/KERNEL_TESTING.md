# DOPPLER Kernel and Pipeline Testing Specification

Defines the testing framework design for WGSL kernels and kernel combinations.

**Implementation status:** See `tests/kernels/README.md`
**Benchmark baselines:** See `tests/kernels/BENCHMARKS.md`

---

## Goals

- Catch kernel correctness regressions early (especially after performance refactors).
- Provide unit tests for individual WGSL kernels.
- Provide integration tests for kernel sequences that occur in real inference.
- Make results interpretable across different GPUs and browsers.

---

## Core Concepts

### Kernel Unit Test

A kernel unit test runs a single WGSL entry point with known inputs and compares outputs to a reference implementation.

### Pipeline Segment Test

A segment test runs a small sequence of kernels that reflects a real subgraph, for example:

- RMSNorm -> QKV matmul -> RoPE -> attention -> residual add
- FFN dense: matmul -> activation -> matmul -> residual add
- MoE: router logits -> softmax+topk -> expert compute -> scatter-add

### Reference Implementation

Use a CPU reference in JavaScript for correctness checks. The reference must:

- Match the math of the WGSL kernel
- Be deterministic
- Clearly define tolerances for floating-point differences

---

## What To Test (Required Coverage)

### Kernels

- Dequantization: Q4_K and MXFP4 paths
- Matmul: f32, f16, mixed precision variants, and the M=1 decode fast path
- RMSNorm
- RoPE
- Attention (all tiers and f16 KV variants)
- Softmax
- Activations: SiLU, GeLU, SwiGLU
- Gather (embedding)
- BiasAdd and ResidualAdd
- TopK and SoftmaxTopK
- MoE kernels: gather and scatter-add

---

## Test Data Strategy

### Deterministic Inputs

Use fixed seeds and explicit arrays rather than random runtime generation.

Recommended patterns:

- Small shapes that fit in one workgroup, and medium shapes that span multiple workgroups.
- Edge shapes: headDim boundaries (64, 128, 256), kvHeads boundaries, and vocab sizes (small mock vocab).
- Numeric edge cases: zeros, large magnitudes, near-underflow values.

### Tolerances

Report tolerances per kernel:

- f32 outputs: default `atol = 1e-4`, `rtol = 1e-4`
- f16 outputs: default `atol = 5e-3`, `rtol = 5e-3`

Attention and softmax tests should check:

- Probability mass sums to approximately 1
- Masking correctness (causal)
- Stability on long sequences (no NaNs, no Infs)

---

## Pipeline Combination Tests (Required)

Define a small set of "golden" segment tests that combine kernels in the same way inference does.

Examples:

1. Dense layer mini-forward (no MoE)
   - Embedding -> RMSNorm -> QKV -> RoPE -> Attention -> Residual -> FFN -> Residual

2. Decode attention step
   - Single-token Q -> cached KV -> attention decode kernel -> residual

3. MoE routing step
   - Router matmul -> softmax+topk -> scatter-add combine

Each segment test should:

- Use small tensor sizes
- Use fixed parameters
- Compare a final output tensor to a CPU reference

---

## Correctness Oracles (Recommended)

When a full CPU reference is expensive:

- Use invariant checks:
  - softmax sum close to 1
  - attention output bounded and finite
  - topk indices sorted by score
- Use cross-implementation checks:
  - compare two GPU variants (streaming attention vs tiled attention) on the same input and confirm close outputs

---

## Debugging Aids (Recommended)

For failures, store:

- WGSL variant name and entry point
- Device capabilities (`shader-f16`, `subgroups`, limits)
- Max absolute error and index of the worst element
- A small excerpt of inputs and outputs

---

## Implementation Layout

| Type | Location | Notes |
|------|----------|-------|
| Kernel tests | `tests/kernels/tests/correctness/` | Unit tests per kernel |
| Benchmarks | `tests/kernels/tests/benchmarks/` | Performance measurement |
| References | `tests/kernels/src/reference/` | CPU reference implementations |
| Harness | `tests/kernels/src/harness/` | Test utilities |
| Browser | `tests/kernels/browser/` | WebGPU test page |

For pipeline segment tests, use `tests/segments/` with CPU refs and saved JSON outputs.

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes, CLI flags (`--kernel-path`, `--kernel-profile`), and the OPFS purge helper.
