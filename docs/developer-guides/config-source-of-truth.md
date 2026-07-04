# Config Source Of Truth

## Goal

Keep conversion, runtime, harness, and kernel configuration from defining the
same behavior in multiple places. This is the front door for answering:

- where does a behavior-changing setting live?
- which file manages WGSL kernel identity?
- which file manages supported and hosted models?
- which checks prove generated mirrors are synchronized?

## Layer Map

| Layer | Owns | Canonical Surface | Generated / Checked Mirrors |
| --- | --- | --- | --- |
| command request | user intent, workload, surface support | `src/tooling/command-api.js`, `src/rules/tooling/command-runtime.rules.json` | CLI/help/browser command checks |
| conversion policy | model-owned inference behavior, quantization, execution graph | `src/config/conversion/**` | stamped `manifest.json` |
| runtime policy | session, storage, debug, benchmark, capability policy | `src/config/schema/*.schema.js`, `src/config/runtime/**` | resolved runtime config |
| rule policy | dtype, kernel, capability, and compatibility selection | `src/rules/**/*.rules.json` | `npm run config:single-source:check` |
| kernel identity | WGSL operation IDs, variant IDs, files, entry points, features, bindings, uniforms, metadata | `src/config/kernels/registry.json` | `src/config/kernels/kernel-ref-digests.js`, reachability reports |
| model registry | supported model metadata, lifecycle, HF coordinates, artifact identity | `models/catalog.json` | quickstart registry, support matrix, support inventory, HF registry |
| benchmark evidence | fairness contract, engine overlays, claim lanes, reports | `benchmarks/vendors/*.json`, `tools/policies/*claim*.json` | generated compare reports and SVG receipts |

## Pluggable Pieces

Add new behavior by plugging into the existing layer that owns that concern:

- new model behavior: conversion config and manifest fields
- new runtime tuning: runtime profile or runtime config payload
- new capability adaptation: rule-map entry or execution graph transform policy
- new WGSL implementation: kernel registry entry plus wrapper and tests
- new hosted model: catalog entry plus verified artifact publication metadata
- new benchmark lane: benchmark policy/profile plus committed evidence artifacts

Do not add a second registry or helper map because a call site is inconvenient.
If an existing surface is hard to use, fix that surface and its checks.

## Ownership Map

| Concern | Canonical Owner |
| --- | --- |
| model dimensions and layer count | conversion config stamped into manifest |
| layer pattern and attention layout | conversion config stamped into manifest |
| activation function and FFN shape | conversion config stamped into manifest |
| quantized storage layout | conversion config and converted shards |
| weight/shard identity | manifest plus shard digests |
| WGSL file, entry point, and digest | conversion `inference.execution` and manifest execution graph |
| runtime session dtype and decode-loop policy | manifest session baseline plus `runtime.inference.session` |
| runtime capability adaptation | execution graph transforms gated by `kernelPathPolicy` |
| kernel variant reachability | `src/config/kernels/registry.json` and rule maps |
| runtime profile metadata | `src/config/runtime/**` profile wrapper fields |
| benchmark fairness knobs | benchmark shared config |
| engine-specific benchmark knobs | benchmark engine overlay |
| command workload and intent | command request and command API normalizers |

## WGSL Kernel Management

Use `src/config/kernels/registry.json` as the single management point for kernel
identity. It owns operation IDs, variant IDs, WGSL filenames, entry points,
feature requirements, binding layout, uniforms, reachability metadata, and
human-readable metadata.

Runtime execution graphs and manifests pin kernels by registry-backed filename,
entry point, and digest. JS wrappers may dispatch only resolved variants from
the registry or rule maps. WGSL files do compute only; they must not encode
selection policy.

Derived kernel assets:

- `src/config/kernels/kernel-ref-digests.js` is generated from the kernel
  registry and WGSL contents.
- `src/gpu/kernels/codegen/wgsl-variants.js` owns generated shader variants.
- reachability metadata is synchronized by `tools/sync-kernel-reachability.js`.

Kernel management checks:

- `npm run kernels:codegen:check`
- `npm run kernels:registry:check`
- `npm run kernels:digests:check`
- `npm run kernels:supported-manifests:report`

## Model Registry Management

Use `models/catalog.json` as the repo source of truth for supported model
metadata: labels, aliases, lifecycle, artifact identity, quickstart/demo
visibility, vendor benchmark mapping, benchmark evidence citations, and hosted
Hugging Face coordinates.

Catalog entries may be unpromoted, but they still require a real conversion
family. Do not add a new-family candidate to the catalog until a checked-in
conversion config exists for that family in the same change or an earlier one.
Consumer priority lists can drive onboarding order; they must not become a
second model registry.

`vendorBenchmark` is the comparable baseline mapping. `benchmarkEvidence` is the
checked-in citation set for a benchmark-selected lane and must point at the
runtime report, compare result, and SVG summary receipts.

Artifact bytes have a separate source of truth:

- external volume: complete local RDRR artifacts and shards
- `models/local/**`: developer-local manifest/tokenizer cache, not a release or
  CI source
- Hugging Face `Clocksmith/rdrr`: published subset generated from approved
  catalog entries

Generated model mirrors:

- `src/client/doppler-registry.json` is a quickstart mirror generated from
  `models/catalog.json`.
- `docs/model-support-matrix.md` and `docs/model-support-inventory.md` are
  generated status views.
- HF `registry/catalog.json` is rebuilt from approved hosted catalog entries.

Model registry checks:

- `npm run support:matrix:check`
- `npm run support:inventory:check`
- `npm run artifact:contract:check`
- `npm run registry:hf:check`
- `npm run ci:catalog:check`

Hosted models are managed by the catalog `hf` block (`repoId`, `revision`,
`path`) plus artifact identity fields. Do not maintain a separate hosted-model
list in CLI, demo, benchmark, or quickstart code.

## Capability-Based Adaptation

Capability adaptation is explicit policy, not fallback behavior.

- Primary execution identity belongs in the conversion config and manifest
  execution graph.
- Hardware adaptation belongs in
  `src/rules/inference/capability-transforms.rules.json` and must match exact
  manifest fields or explicit variant IDs.
- Runtime profiles may request an experimental lane, but unsupported
  capabilities must fail closed or apply a named transform that is visible in
  resolved execution state.
- JS may inspect device capabilities only to feed rule selection or validate a
  resolved contract. It may not silently choose a different model behavior.

## Rules

- Conversion config is the source for model-owned inference behavior. Runtime
  config may not rediscover or rewrite it.
- Runtime profiles tune execution policy only. They must not carry string
  `runtime.inference.kernelPath` registry IDs.
- `runtime.inference.kernelPath` is either `null` or an inline object generated
  from execution-v1. String registry IDs are legacy and must fail fast.
- `src/config/kernels/registry.json` is the operation-kernel registry used by
  kernel wrappers and reachability tooling. It is not a model kernel-path
  registry.
- Non-profile policy assets under `src/config/runtime/**` must declare their own
  schema and be explicitly allowlisted by checks.
- Mirrors are allowed only when generated or check-synced from the canonical
  owner.
- Documentation may describe generated mirrors, but must name the canonical
  owner and the check command beside each mirror.
- If a behavior choice cannot be traced to one row in the layer map, the change
  is not ready to merge.

## Forbidden Duplication

- No model metadata copied into benchmark scripts, demo registries, or CLI
  aliases outside catalog-generated mirrors.
- No WGSL filename suffix grammar outside `src/config/kernels/registry.json`
  and its sync tooling.
- No hidden dtype, kernel, session, or generation defaults in runtime JS.
- No model-family substring matching in runtime code or rule maps.
- No hosted HF registry edits that bypass `models/catalog.json` and the publish
  tooling.

## Program Bundle Export Implication

The Doppler Program Bundle references the manifest execution graph, WGSL
digests, runtime/capture profile, and artifact identities from these owners.
It must not invent a Doe-specific duplicate list of model behavior, kernels, or
weight layout. See `docs/integration/program-bundle.md`.
