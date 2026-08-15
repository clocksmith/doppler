# CATSCAN: Inference Runtime

Component: `doppler.runtime-source.inference`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Turn validated model and runtime contracts into deterministic semantic plans, execution, and model outputs.

## Authority

- Owns inference orchestration, runtime model contracts, semantic plans, KV-cache coordination, sampling, and execution receipts.
- Does not own artifact conversion, storage bytes, raw GPU math, or application intent.

## Scope

- Mainline inference mechanisms, pipeline registry, cache behavior, and execution adapters.

## Contracts

- Input: [Runtime model contract](runtime-model.js), resolved configuration, loaded tensors, and normalized commands.
- Output: Generated tokens, embeddings, rankings, model state transitions, and behavior receipts.

## Invariants

- Command, session, semantic plan, bound plan, and executor remain distinct.
- Adapter differences cannot introduce hidden numerical choices.
- Resource ownership and first failure boundaries remain observable.

## Acceptance

- Inference boundaries, behavioral parity, lifecycle, and integration tests pass.
- Evidence: [inference tests](../../tests/inference).

## Non-goals

- Application orchestration or silently broadening one model's evidence to a family.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
