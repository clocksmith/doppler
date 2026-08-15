# CATSCAN: Configuration Contracts

Component: `doppler.runtime-source.config`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Resolve all runtime-visible policy from validated, traceable, single-source configuration contracts.

## Authority

- Owns schemas, config registries, merge rules, runtime profiles, support tiers, and resolved configuration transforms.
- Does not own numeric execution, request intent, or conversion facts after artifact materialization.

## Scope

- Configuration code and assets under `src/config/`.

## Contracts

- Input: [Config contract](../../docs/config.md), manifests, runtime overlays, platform capabilities, and policy assets.
- Output: Validated resolved configuration with source attribution and explicit errors.

## Invariants

- Required values are never recreated as hidden runtime defaults.
- Nullable disabled state remains distinct from missing state.
- Runtime overlays cannot silently rewrite conversion-owned facts.

## Acceptance

- Config single-source, schema, merge, and contract tests pass.
- Evidence: [configuration tests](../../tests/config).

## Non-goals

- Executing kernels or using configuration files as mutable runtime state.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
