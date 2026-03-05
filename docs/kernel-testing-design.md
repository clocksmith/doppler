# Kernel Testing Design

Design guidance for WGSL kernel correctness and composition tests.

## Goals

- catch kernel regressions early
- validate individual kernels and representative kernel sequences
- keep results interpretable across GPU/browser variants

## Core concepts

### Kernel unit test

Run one WGSL entry point with known inputs and compare to deterministic CPU reference.

### Pipeline segment test

Run a short real sequence, for example:
- RMSNorm -> QKV -> RoPE -> attention -> residual
- FFN dense path
- MoE routing + combine path

### Reference implementation

Reference checks must be deterministic and explicit about tolerance.

## Required coverage

- dequantization paths (Q4_K, MXFP4 where applicable)
- matmul variants (f16/f32/mixed)
- RMSNorm, RoPE, attention, softmax
- activations, gather/residual, topk/softmax-topk
- MoE gather/scatter kernels

## Data strategy

- fixed seeds and explicit arrays
- small and medium tensor shapes
- edge numeric cases (zero/large/small values)

Default tolerance guidance:
- f32: `atol=1e-4`, `rtol=1e-4`
- f16: `atol=5e-3`, `rtol=5e-3`

## Debug payload for failures

Capture:
- kernel/entrypoint
- device capability flags
- max absolute error and index
- input/output snippets

## Implementation map

- kernel harness: `tests/kernels/harness/`
- browser tests: `tests/kernels/browser/test-page.js`
- references: `tests/kernels/reference/`

## Related

- Runbook: [testing-runbook.md](testing-runbook.md)
- Kernel tests: [../tests/kernels/README.md](../tests/kernels/README.md)
