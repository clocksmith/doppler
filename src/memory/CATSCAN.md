# CATSCAN: Memory Ownership

Component: `doppler.runtime-source.memory`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Provide explicit, reusable GPU and host-memory ownership mechanisms with measurable limits and cleanup.

## Authority

- Owns buffer pooling, host heaps, address tables, memory capability detection, and allocation accounting.
- Does not own kernel selection, model policy, or storage persistence.

## Scope

- Shared host and GPU memory-management primitives under `src/memory/`.

## Contracts

- Input: Requested byte sizes, lifecycle scopes, device capabilities, and resolved memory policy.
- Output: Owned allocations, reusable buffers, capability facts, and memory statistics.

## Invariants

- Acquisition and release ownership are explicit.
- Failure paths do not leak mapped, pooled, or temporary resources.
- Capability facts constrain behavior but do not invent policy.

## Acceptance

- Pool, heap, alias, cleanup, and capability tests pass.
- Evidence: [memory tests](../../tests/memory).

## Non-goals

- Deciding precision, cache layout, or application memory budgets implicitly.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
