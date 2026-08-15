# CATSCAN: Experimental Lanes

Component: `doppler.runtime-source.experimental`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Contain non-core capabilities so they can evolve without being mistaken for product-supported runtime behavior.

## Authority

- Owns quarantined adapters, training, distribution, hotswap, diffusion, energy, browser conversion, bridge, and orchestration implementations.
- Does not own Tier 1 claims or bypass the same artifact, identity, and evidence contracts as mainline code.

## Scope

- All code under `src/experimental/` unless promoted through an explicit boundary change.

## Contracts

- Input: [Subsystem support registry](../config/support-tiers/subsystems.json) and experiment-specific contracts.
- Output: Explicit experimental capabilities, candidates, and retained positive or negative evidence.

## Invariants

- Directory presence and exports do not imply support.
- Promotion changes support policy and acceptance evidence in the same change.
- Experimental failures remain explicit and reproducible.

## Acceptance

- Subsystem support and relevant lane-specific contract checks pass.
- Evidence: [generated subsystem matrix](../../docs/subsystem-support-matrix.md).

## Non-goals

- A permanent dumping ground or a shortcut around product qualification.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
