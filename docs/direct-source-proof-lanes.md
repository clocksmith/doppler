# Direct-source proof lanes

Canonical promotion plan for turning persisted direct-source artifacts
(`safetensors`, `gguf`) into additional proof-grade runtime lanes without
displacing the current `RDRR` production path.

## Current state

Direct-source runtime inputs already exist in Doppler. Raw-source artifacts are
normalized into runtime-model contracts and can be persisted as direct-source
manifests. They are still an experimental subsystem tier and are not part of
the canonical quickstart or demo proof path.

`RDRR` remains the primary runtime artifact for:

- the verified quickstart and hosted model path;
- the current production Doppler artifact contract;
- the current Doe/Cerebras Gemma-4 portability proof lane.

This document is about additional canonical proof lanes. It does not replace
`RDRR`, and it must not weaken the current `RDRR`-first proof story.

## Why add direct-source proof lanes

Additional canonical direct-source lanes are useful because they prove a
different class of claims than `RDRR`:

- `safetensors` is the clearest upstream-checkpoint fidelity lane.
- `gguf` is the clearest compatibility and ecosystem-ingest lane.
- both formats test whether Doppler's runtime-model planning can stay faithful
  to a source artifact without requiring a conversion step first.

The direct-source lanes should therefore become additive proof surfaces:

- `RDRR` remains the production Doppler artifact lane.
- `safetensors` becomes the upstream raw-checkpoint lane.
- `gguf` becomes the compatibility-format lane.

## Hard rules

### Additive, not replacing

Direct-source proof lanes must be introduced as additional canonical lanes.
They do not demote or replace `RDRR` as the production runtime artifact.

### Persisted manifest required

No direct-source lane is proof-grade until it uses a persisted materialized
direct-source manifest with artifact-relative paths and complete digests.
Ad hoc local paths, synthetic bundles, or incomplete digest coverage are
debugging aids only.

### Same receipt discipline

Direct-source lanes become canonical only when they can satisfy the same class
of evidence expected from `RDRR`:

- stable source artifact identity;
- stable runtime-model / execution identity;
- deterministic reference transcript;
- bounded parity receipts;
- fail-closed runtime behavior.

### No format-specific runtime branch as the proof

The proof target is not "Doppler can parse this file format." The proof target
is "this source artifact normalizes into the same governed runtime-model
contract shape and survives the same transcript and parity gates."

## Proof-grade contract surface

### Persisted direct-source manifest

A proof-grade direct-source lane still starts from a materialized
`manifest.json`. The manifest is the portable artifact boundary. Raw
`safetensors` or `gguf` files are source assets behind that manifest, not a
second execution contract.

At minimum, the materialized manifest must stabilize these
`metadata.sourceRuntime` fields:

- `mode`, `schema`, `schemaVersion`, `sourceKind`, `hashAlgorithm`, and
  `pathSemantics`;
- `sourceFiles[]` with stable `index`, `path`, `size`, `hash`, and
  `hashAlgorithm`;
- `auxiliaryFiles[]` with stable `path`, `size`, `hash`, `hashAlgorithm`, and
  `kind`;
- tokenizer asset paths under `tokenizer.jsonPath`, `tokenizer.configPath`,
  and `tokenizer.modelPath` when those assets exist.

For a canonical proof lane, `pathSemantics` must be `artifact-relative`.
`runtime-local` manifests are local bring-up aids only and are not portable
proof inputs.

If source-file digests, auxiliary-file digests, or tokenizer/config identity
coverage are incomplete, the lane remains experimental.

### Runtime-model contract

The persisted direct-source manifest must normalize into a stable
runtime-model contract. The proof-grade surface is not just the raw manifest
metadata; it is the normalized model contract Doppler actually executes.

At minimum, the normalized runtime-model contract must keep these fields
stable for the same source artifact:

- `kind`, `sourceFormat`, `version`, `modelId`, `modelType`, `hashAlgorithm`;
- `quantization` plus `quantizationInfo`;
- `architecture`;
- `inference`, including the declared execution graph;
- `shards`, `totalSize`, `tensorCount`, `tensors`, and `groups`;
- `metadata.sourceRuntime`.

Execution identity must still come from the normalized execution graph, not
from raw file-format heuristics. Converter-style quantization overrides remain
out of bounds for direct-source proof lanes; a quantized variant that changes
runtime semantics requires a real convert step that emits a new artifact.

### Program Bundle binding

Promotion to a portable Doe/Cerebras proof lane happens only after the
direct-source manifest can be exported through the normal Doppler Program
Bundle path.

That means a proof-grade direct-source lane must bind:

- the exact materialized manifest hash;
- the normalized execution graph hash;
- the normalized weight/artifact identity set;
- the bounded reference transcript;
- the source-runtime identity carried by the materialized manifest.

If a proposed direct-source lane needs Doe to ingest raw source files directly
instead of the closed bundle, it is not yet promotion-ready.

## Promotion order

### Phase 1: Safetensors as the first additional canonical lane

Promote `safetensors` first because it has the cleanest checkpoint provenance
story and the fewest imported runtime assumptions.

Required gates:

- persisted direct-source manifest with complete digests for source files and
  auxiliary tokenizer/config assets;
- canonical runtime-model contract emitted from the persisted manifest;
- explicit architecture/config normalization with fail-closed rejection on
  missing required fields;
- deterministic bounded reference run with prompt identity, generated token
  identity, and transcript hashes;
- operator-boundary diagnostics good enough to isolate divergence by semantic
  stage, not just by file format offset;
- publication and reload flow proven through the same hosted/local artifact
  roots used by the current proof lanes.

Non-goals for this phase:

- replacing `RDRR` in quickstart flows;
- converter-style quantization overrides on direct-source loads;
- special-casing Doe/Cerebras around raw source format details.

### Phase 2: Doe acceptance of direct-source proof bundles

Once Doppler has a proof-grade `safetensors` direct-source lane, Doe can admit
that lane as an additional portable-program input.

Required gates:

- the direct-source lane exports bundle identity fields that Doe can bind into
  ingest receipts without inventing a second source of truth;
- Doe ingest accepts the normalized portable-program facts, not raw
  format-specific runtime assumptions;
- Doe WebGPU and Doe-emitted CSL both bind to the same source artifact
  identity, normalized execution identity, prompt/input contract, and transcript
  contract;
- simulator parity lands before any hardware promotion.

`RDRR` remains the primary Cerebras lane while this work is incomplete.

### Phase 3: GGUF as the compatibility lane

Promote `GGUF` after the `safetensors` lane is stable.

Additional gates beyond the safetensors lane:

- explicit mapping from GGUF metadata to normalized runtime-model config;
- explicit documentation of every imported assumption from GGUF metadata and
  every rejected one;
- explicit proof that quantization/storage conventions do not silently rewrite
  execution semantics;
- a clear separation between "GGUF compatibility proof" and "production
  Doppler artifact proof."

The burden of proof is higher for `GGUF` because its ecosystem conventions are
more runtime-opinionated than raw `safetensors`.

## What must become canonical before promotion

These fields and behaviors must be stable before a direct-source lane can be
called canonical:

- persisted direct-source manifest shape;
- source file identity and digest semantics;
- auxiliary tokenizer/config identity coverage;
- runtime-model contract fields derived from direct-source planning;
- transcript export and parity schema;
- publication and cache-reopen workflow.

If any of those are still experimental, the lane remains experimental.

## Review question

Use this review question for any work in this area:

> Did this make a direct-source lane more governed and receiptable, or did it
> bypass the governed artifact contract?

If the change bypasses the governed artifact contract, it is not promotion work.

## Relationship to other docs

- [RDRR Format](rdrr-format.md): canonical `RDRR` artifact contract.
- [Conversion Runtime Contract](conversion-runtime-contract.md): conversion-owned
  vs runtime-owned field boundaries, including direct-source adaptation rules.
- [Architecture](architecture.md): runtime-model construction and subsystem-tier
  placement.
- [Promote a Verified Runtime Artifact](developer-guides/05-promote-model-artifact.md):
  operational workflow for publishing direct-source manifests or `RDRR`
  artifacts.
