# CATSCAN: WebGPU Compute

Component: `doppler.runtime-source.gpu`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Execute fully resolved tensor operations on WebGPU with deterministic resource and submission ownership.

## Authority

- Owns device lifecycle, GPU command mechanics, kernel dispatch, readback, submission tracking, and GPU-facing tensor resources.
- Does not own model intent, artifact facts, or hidden dtype and kernel selection policy.

## Scope

- WebGPU mechanisms under `src/gpu/`, excluding host-memory policy owned by memory.

## Contracts

- Input: Resolved operation plans, [rule contracts](../rules/CATSCAN.md), buffers, and device capabilities.
- Output: Submitted GPU work, owned resources, readbacks, and execution observations.

## Invariants

- GPU objects enter only after semantic choices are resolved.
- Every acquired resource has one explicit cleanup path.
- Device loss and unsupported capability remain visible failures.

## Acceptance

- GPU contract, resource cleanup, and kernel suites pass.
- Evidence: [GPU tests](../../tests/gpu).

## Non-goals

- Choosing application policy or inferring model families at dispatch time.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
