# CATSCAN: Electron release adapter

Component: `doppler.runtime-source.client.electron`

Parent: [Client API](../CATSCAN.md)

## Target

Bind an Electron renderer to an eligible immutable Pack while keeping durable update state and customer activation authority in the main process.

## Authority

- Owns the typed IPC contract, atomic release-state transitions, and renderer Pack-session coordination.
- Does not own hardware qualification, eligibility signing, customer activation authority, or tensor mathematics.

## Scope

- Electron main-process release state, renderer Pack sessions, and their typed IPC boundary.

## Contracts

- Input: verified release decisions, verified revocation snapshots, customer activation digests, and Pack references.
- Output: durable current/previous/candidate state and renderer sessions opened only from a current usable Pack.

## Invariants

- Candidate installation never changes the active Pack.
- Eligibility and explicit customer authorization are both required for activation.
- Rejected upgrades preserve the current and previous Packs.
- Expired or missing signed revocation state fails closed before a Pack is opened.

## Acceptance

- Electron release-state and renderer-adapter contract tests pass.
- Evidence: [Electron client tests](../../../tests/client).

## Non-goals

- Owning Electron lifecycle, application deployment, customer updater policy, or production key custody.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
