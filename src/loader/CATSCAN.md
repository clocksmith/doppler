# CATSCAN: Tensor Loader

Component: `doppler.runtime-source.loader`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Materialize validated artifact tensors into runtime-owned CPU and GPU representations with bounded memory behavior.

## Authority

- Owns shard/tensor loading coordination, dtype materialization, dequantization, caching, and loader resource lifecycle.
- Does not own manifest facts, storage backend policy, or inference selection.

## Scope

- Model loaders, tensor loaders, shard resolution, load-time caches, and weight preparation.

## Contracts

- Input: Validated [RDRR contracts](../formats/rdrr/CATSCAN.md), storage contexts, and resolved loading policy.
- Output: Loaded weights and explicit load timing through [loader exports](index.js).

## Invariants

- Loader code never mutates inference configuration.
- Materialized dtype and source transforms remain inspectable.
- Partial loads and failures clean up owned resources.

## Acceptance

- Loader, dequantization, shard, memory, and failure-path tests pass.
- Evidence: [loader tests](../../tests/loader).

## Non-goals

- Choosing model behavior or hiding missing tensors with runtime defaults.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
