# CATSCAN: Configuration Contracts

Component: `doppler.runtime-source.config`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Resolve runtime policy from validated, traceable configuration contracts.

## Authority

- Owns schemas, registries, merges, profiles, support tiers, and configuration resolution.
- Executable graph transformations belong to Rig under `src/converter/`; historical transform imports are compatibility exports only.
- Does not own numeric execution, request intent, or conversion facts after artifact materialization.

## Scope

- Code and assets under `src/config/`.

## Contracts

- Invalid configuration uses shared `errors` envelopes; config retains policy authority.

- Input: [Config contract](../../docs/config.md), manifests, runtime overlays, platform capabilities, and policy assets.
- Output: Validated configuration, sources, and errors.

## Invariants

- Immutable registry data and identified companion validators bind to accepted execution identity.

- Required values are never recreated as hidden runtime defaults.
- Nullable disabled state remains distinct from missing state.
- Runtime overlays cannot silently rewrite conversion-owned facts.
- Capsule v3 executable identity excludes release events; Capsule v2 and v3 use capsule-namespaced schemas. Pre-migration formats are rejected, never relabeled during opening.
- Signed release history must satisfy explicit trust, time, and persisted-checkpoint policy.
- Recipient-pinned local use may outlive managed eligibility only through an explicit acceptance record acknowledging unseen revocations. It cannot waive known denials or authorize delegation.

## Acceptance

- Config single-source, schema, merge, and contract tests pass.
- Evidence: [configuration tests](../../tests/config).

## Non-goals

- Executing kernels or using configuration files as mutable runtime state.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
