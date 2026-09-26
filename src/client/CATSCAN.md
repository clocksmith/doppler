# CATSCAN: Client API

Component: `doppler.runtime-source.client`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Invoke qualified models while preserving execution identity.

## Authority

- Owns public validation, source coordination, model handles, and observations.
- Does not own inference, artifact facts, or silent provider/precision substitution.

## Scope

- Client facades, providers, receipts, and runtime coordination.

## Contracts

- Input: [Model catalog](../../models/CATSCAN.md), public requests, artifact contracts, and runtime configuration.
- Output: Loaded model handles, generated or embedded results, and inspection receipts.

## Invariants

- Requests honor artifact/execution identities, pins, and authorized alternatives.
- Declared hosts preserve facade parity. Exports use Capsule names; former Pack names/subpaths are not aliases.
- `doppler-gpu/run` and `createDopplerRun` retain the runtime names as aliases.
- Execution consumes verified bytes, never unverified refetches, and reports selected-plan/artifact-closure evidence.
- V3 receipts distinguish managed eligibility from retained use. Verified denials advance durable checkpoints; unauthenticated histories cannot.
- `doppler-gpu/host` composes existing ports without choosing trust, upgrades, or bypassing checks.
- Existing observers emit throttled bytes and immediate acquisition/hash/verification/preparation transitions; verified completion requires size/digest success, reuse is honest.
- Progress preserves private ownership, host-task yielding, cancellation, and verification; it never authorizes execution.

## Acceptance

- Client contract, provider, and root-facade tests pass.
- [Local-search progress acceptance](../../docs/goals.md#next-product-increment-copy-and-run-local-search) requires corruption, reuse, cancellation, and responsive-task tests; requirements are not completion claims.
- Evidence: [client tests](../../tests/client).

## Non-goals

- GPU math, conversion, application trust, parsing, search/index policy, and application controllers.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
