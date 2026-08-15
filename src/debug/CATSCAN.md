# CATSCAN: Runtime Observation

Component: `doppler.runtime-source.debug`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Expose structured runtime observation without silently changing execution or evidence meaning.

## Authority

- Owns logging, tracing, probes, capture policy, diagnostic statistics, and observation signals.
- Does not own runtime selection, benchmark acceptance, or user-interface presentation.

## Scope

- Shared runtime debug and inspection primitives.

## Contracts

- Input: [Observation policy registry](../config/inspection/observation-policies.json) and emitted runtime events.
- Output: Structured logs, traces, probes, statistics, and declared observation effects.

## Invariants

- Observation that changes execution is labeled and excluded from representative timing.
- Runtime code uses the shared debug system rather than ad hoc logging.
- Evidence records first failure boundaries without conflating them with causes.

## Acceptance

- Debug policy, receipt, and runtime observation tests pass.
- Evidence: [debug-related inference tests](../../tests/inference).

## Non-goals

- Choosing execution policy or turning diagnostic output into a support claim.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
