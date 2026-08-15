# CATSCAN: Client API

Component: `doppler.runtime-source.client`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Make qualified local model workloads easy to invoke while preserving exact resolved execution identity.

## Authority

- Owns the root application facade, public call-shape validation, source resolution coordination, and inspectable model handles.
- Does not own inference algorithms, artifact facts, or silent provider and precision substitution.

## Scope

- Browser and Node client facades, provider integration, receipts, and runtime coordination.

## Contracts

- Input: [Model catalog](../../models/CATSCAN.md), public requests, artifact contracts, and runtime configuration.
- Output: Loaded model handles, generated or embedded results, and inspection receipts.

## Invariants

- Logical requests resolve to visible artifact and execution identities.
- Exact pins and policy-authorized alternatives are honored.
- Public facade behavior remains aligned across declared hosts.

## Acceptance

- Client contract, provider, and root-facade tests pass.
- Evidence: [client tests](../../tests/client).

## Non-goals

- Owning GPU math, model conversion, or application trust decisions.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
