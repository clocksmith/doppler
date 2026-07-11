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

## Reduction and tail invariants

Correctness properties that can fail at runtime belong in executable GPU tests,
not only in kernel comments or performance notes.

For any one-phase or two-phase argmax kernel, test all of the following:

- duplicate maximum values placed in different workgroups select the lower
  vocabulary index
- both phases use the same `(value, index)` comparison rule
- the reduction identity is `(-inf, UINT_MAX)`
- a vocabulary length shaped as `tile * N + 3` cannot select a padded lane
- padded values are `-inf`, not zero
- an unsupported subgroup width rejects the specialized variant instead of
  silently changing its row packing

For a 16-lane subgroup row reduction, add a source or execution-contract check
that the reduction uses only shuffle offsets `1`, `2`, `4`, and `8`. Also keep a
CPU-reference numeric test with f32 accumulation and an exact 128-token greedy
golden-parity run at the model/profile level.

Rejected experimental kernels do not need to remain in the registry, but these
tests are required before the same reduction design can be promoted later.

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
- Kernel tests: [../tests/kernels/GUIDE.md](../tests/kernels/GUIDE.md)
