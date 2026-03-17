# Doppler Testing

This page is now a testing index.

## Canonical docs

- How to run tests: [testing-runbook.md](testing-runbook.md)
- Kernel test design: [kernel-testing-design.md](kernel-testing-design.md)
- Harness reference: [../tests/README.md](../tests/README.md)
- Kernel coverage: [../tests/kernels/README.md](../tests/kernels/README.md)
- Kernel baselines: [../tests/kernels/benchmarks.md](../tests/kernels/benchmarks.md)

## Inference matrix strategy

See [inference-test-matrix.md](inference-test-matrix.md) for how Doppler slices the combinatorial explosion of model families, quantization formats, attention modes, batch sizes, and decode paths without maintaining an intractable full cross-product.

## Contract note

Behavior is driven by JSON contracts plus runtime config. WGSL is arithmetic execution only.

## Kernel override note

For override and compatibility policy, use the canonical section in [operations.md#kernel-overrides--compatibility](operations.md#kernel-overrides--compatibility).
