# CATSCAN: Pipeline Registry

Component: `doppler.runtime-source.inference.pipelines`

Parent: [Inference Runtime](../CATSCAN.md)

## Target

Bind declared model types to pipeline implementations without changing normalized command or session semantics.

## Authority

- Owns pipeline registration, construction, shared pipeline contexts, and pipeline-family boundaries.
- Does not own model catalog support state, artifact parsing, or adapter-local numeric policy.

## Scope

- Pipeline registry, factories, shared contexts, and declared pipeline families.

## Contracts

- Input: Validated manifests, resolved runtime contexts, and registered pipeline modules.
- Output: Initialized pipeline instances through the [pipeline registry](registry.js).

## Invariants

- Unknown model types fail explicitly.
- Pipeline selection is declared before execution.
- Shared command and session semantics survive pipeline-family dispatch.

## Acceptance

- Pipeline registry, factory, context, and workload tests pass.
- Evidence: [pipeline integration tests](../../../tests/integration).

## Non-goals

- Runtime model-name heuristics or a universal pipeline with hidden workload branches.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
