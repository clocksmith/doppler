# CATSCAN: GPU Kernels

Component: `doppler.runtime-source.gpu.kernels`

Parent: [WebGPU Compute](../CATSCAN.md)

## Target

Implement registered deterministic tensor operations whose numeric behavior and dispatch identity can be verified.

## Authority

- Owns WGSL math, kernel wrappers, bind layouts, pipeline construction, and interpretation of resolved kernel specs.
- Does not own model-family detection, runtime fallback policy, or benchmark promotion.

## Scope

- Registered kernel JavaScript, WGSL sources, generated shader assets, and kernel-local execution helpers.

## Contracts

- Input: [Kernel registry](../../config/kernels/registry.json), resolved specs, and typed buffers.
- Output: Deterministic operator results and kernel execution identity.

## Invariants

- Registry IDs, files, entrypoints, bindings, and digests remain aligned.
- Immediate and recorded adapters consume the same semantic plan.
- Numerical changes pass operator and continuation evidence appropriate to the change.

## Acceptance

- Kernel registry, digest, browser, and reference parity checks pass.
- Evidence: [kernel test suite](../../../tests/kernels).

## Non-goals

- Selecting workloads, authoring runtime defaults, or claiming end-to-end model quality.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
