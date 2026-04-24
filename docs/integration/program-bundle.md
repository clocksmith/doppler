# Doppler Program Bundle

This document defines the Doppler-owned export shape for a portable model
program. A Program Bundle is the closed artifact that browser WebGPU, Node
providers, Doe.js capture, and Doe backend lowering should consume.

The bundle is not arbitrary JavaScript. It is declared model identity,
manifest/config state, constrained JS host orchestration, declared WGSL
kernels, artifact identities, and validation transcripts tied together by
hashes.

This is the normative home for Program Bundle fields. Doe owns ingest,
capture, HostPlan, CSL, and receipt validation in its own repo; Ouroboros owns
only the cross-repo journey and map.

## Purpose

The Program Bundle is Doppler's portable model-program artifact for downstream
execution backends such as Doe/Cerebras. It names a constrained Doppler program
made from:

- a materialized `manifest.json` for either an `RDRR` artifact or a persisted
  direct-source artifact
- the manifest-owned execution-v1 graph
- the reachable WGSL kernel closure
- declared JS host entrypoints
- model artifacts and hashes
- a browser/WebGPU reference transcript

Doppler owns the model program. Doe owns capture, HostPlan generation,
WGSL-to-CSL lowering, and backend execution.

```text
Doppler authoring layer
  -> Doppler Program Bundle
       -> browser WebGPU reference
       -> Node WebGPU / Dawn / Doe.js provider run
       -> Doe ingest for HostPlan / CSL lowering
```

Doe may execute the JS entrypoint through a provider for capture and
validation, but portable lowering is based on the declared execution-v1 graph,
declared WGSL modules, and declared artifacts.

In one sentence: the Program Bundle is the single source of truth; Doppler
executes it on WebGPU, and Doe lowers the same declared program into backend
artifacts such as HostPlan and CSL.

## Contract

Schema: `doppler.program-bundle/v1`

Implementation:

- schema and validator: `src/config/schema/program-bundle.schema.js`
- Node exporter/checker API: `src/tooling/program-bundle.js`
- CLI wrapper: `doppler program-bundle`
- tools: `tools/export-program-bundle.js`, `tools/check-program-bundle.js`
- bounded reference-run exporter: `tools/run-program-bundle-reference.js`
- parity checker: `tools/check-program-bundle-parity.js`
- example: `examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json`

The bundle must contain:

| Field | Meaning |
| --- | --- |
| `sources.manifest.hash` | SHA-256 of the exact manifest JSON |
| `sources.executionGraph.hash` | Stable hash of `manifest.inference.execution` |
| `sources.executionGraph.expandedStepHash` | Stable hash of expanded execution-v1 steps |
| `sources.weightSetHash` | Stable hash of declared shard paths, sizes, and digests |
| `sources.artifactSetHash` | Stable hash of every declared artifact in the bundle |
| `host` | constrained JS host contract |
| `wgslModules` | tree-shaken reachable WGSL module closure |
| `execution.kernelClosure` | declared/reachable/excluded kernel IDs and phase counts |
| `execution.steps` | expanded execution steps with kernel IDs, symbolic dispatch, and static binding metadata |
| `captureProfile` | deterministic capture surfaces, adapter stamp, and capture hash |
| `artifacts` | manifest, shard, tokenizer, conversion config, and reference report hashes |
| `referenceTranscript` | prompt/output/token/phase/KV identity from a browser report |

The bundle must be closed: no hidden imports, no implicit kernels, no
runtime-discovered shader strings, and no undeclared model behavior.
Unsupported features must reject early when `unsupportedFeaturePolicy` is
`fail`; there is no downgrade from portable program to best-effort capture.

## Direct-source manifests

Program Bundle v1 stays manifest-first even when the source artifact began life
as `safetensors`, `gguf`, or another direct-source format.

For a persisted direct-source artifact:

- `sources.manifest.hash` still names the exact materialized `manifest.json`;
- raw source-file identity enters through `manifest.metadata.sourceRuntime` and
  the declared artifact set, not through an undeclared raw-file side channel;
- `sources.executionGraph.hash` and `execution.steps` still come from the
  normalized execution-v1 graph, not from source-format-specific runtime
  assumptions.

For proof-grade direct-source bundles, the materialized manifest must carry
portable source-runtime identity:

- `metadata.sourceRuntime.pathSemantics` must be `artifact-relative`;
- `metadata.sourceRuntime.sourceFiles[]` and `auxiliaryFiles[]` must carry
  complete digest coverage;
- tokenizer/config assets required for replay must be declared through the same
  materialized-manifest contract.

If a direct-source input still depends on `runtime-local` paths, incomplete
digests, or loader-only assumptions that do not survive the materialized
manifest, it is not yet a portable Program Bundle source.

## Authoring Sources

Doppler authoring remains in normal project-owned locations:

- `src/config/conversion/` for conversion-time artifact and execution-v1 policy
- `src/config/runtime/` for runtime profiles
- `src/models/` for model registration
- `src/inference/` for JS orchestration
- `src/gpu/` for WGSL kernels and WebGPU dispatch
- `src/rules/` for rule-map policy

The WGSL variant registry under `src/config/kernels/registry.json` is metadata
for WGSL variants. It is not a string kernel-path ID registry. Portable
program identity comes from execution-v1 graph steps and pinned WGSL digests.

## JS Host Boundary

The host boundary is intentionally narrow:

- `host.jsSubset` must be `doppler-webgpu-host/v1`.
- declared entrypoints name Doppler JS modules and exports.
- dynamic imports are disallowed for the model path.
- each declared entrypoint is source-hashed and scanned by the exporter.
- filesystem and network access are limited to declared artifacts.
- DOM access is disallowed in the model execution path.

Portable host JS may:

- build or run the declared model program
- allocate buffers and bind weights
- issue WebGPU-compatible commands
- select prefill/decode behavior from fixed config
- produce or validate a capture trace
- stay deterministic under fixed config and inputs

Portable host JS must not:

- perform hidden filesystem or network work during capture
- import undeclared model execution modules dynamically
- generate undeclared WGSL at runtime
- depend on DOM state for model execution
- mutate global state that changes graph shape
- call provider-specific native APIs unless the bundle marks the path as
  non-portable

This is a tree-shaking/capture contract, not a general JS compiler target.

## WGSL Closure

The exporter reads the manifest execution graph, expands execution-v1, collects
reachable kernel refs, and emits only the reachable WGSL modules. Each module
must have:

- kernel ID
- WGSL file
- entry point
- `sha256:<64 hex>` content digest
- source path when the local registry can resolve it
- static `@group/@binding` declarations
- `@workgroup_size` metadata
- WGSL `override` declarations
- subgroup requirement flag

The exporter compares execution graph digests against the generated kernel
digest registry. A mismatch fails before a bundle is written.

## Execution Metadata

The exporter expands `manifest.inference.execution` and emits
`execution.steps`. Each step records:

- stable step ID and index
- op, phase, section, and layer coverage
- `src`/`dst` state roles
- kernel ID, WGSL file, entry point, and digest
- weight template, constants, and precision metadata
- symbolic dispatch identity and the static WGSL binding set for the module

Concrete dispatch counts still belong to runtime/capture or Doe lowering. The
bundle gives Doe the deterministic static plan and hash-stable closure needed
to reject undeclared kernels before lowering.

## Kernel Identity

Execution-v1 is the bridge. A Program Bundle must not depend on:

- string `runtime.inference.kernelPath` IDs
- legacy `kernelPlan`
- removed kernel-path registry assets
- implicit defaults that change execution behavior

Runtime `kernelPath` may be `null` or an inline object generated from
execution-v1. `null` means an explicit no-override/reset value and must not be
rewritten into defaults.

## Reference Transcript

Program Bundle validation requires a reference transcript. A manual receipt is
not enough.

The exporter accepts a browser/debug report and derives:

- prompt identity and hash
- output text hash
- generated token ID hash
- generated text hash
- per-token proof hashes
- prefill/decode timing and token counts
- stop reason and stop token ID when available
- KV-cache layout, seqLen/maxSeqLen, byte counters, and state hash
- optional KV-cache byte digests by layer/key/value
- explicit logits policy and optional per-step logits digests

Program Bundle reference runs request proof-grade browser diagnostics. Those
reports include `metrics.referenceTranscript` with prompt token identity, full
generated token IDs, per-step finalized logits digests, and KV byte digests for
the used cache region when the active KV layout supports byte readback. Older
reports can still be used only when their token diagnostics are complete enough
to identify the generated output; missing logits remain represented honestly as
`logits.mode: "not-captured"`.

## Validation

A bundle is not proof by itself. It becomes evidence only when a run binds:

- the same manifest hash
- the same execution graph hash
- the same weight/artifact hashes
- the same tokenized input
- the same declared WGSL digests
- a transcript from the selected runtime or backend

For the Cerebras lane, the expected proof target is bounded deterministic
prefill+decode transcript parity, not a single logits tensor snapshot.

## CLI

### One-command composition: `doppler bundle`

```bash
node src/cli/doppler-cli.js bundle \
  --manifest models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json \
  --out reports/program-bundles/gemma-3-270m-it-q4k-ehf16-af32/ci/ \
  --prompt "The color of the sky is" \
  --max-tokens 8 \
  --surface browser
```

`doppler bundle` is the WS1 exit-condition command. It chains the four
stages that previously required separate invocations:

| Stage | Emits | Schema |
| --- | --- | --- |
| intake | `intake-report.json` | `doppler.intake-report/v1` |
| capture | `reference-report.json`, `reference-transcript.json` | verify report; `doppler.reference-transcript/v1` |
| bundle | `program-bundle.json` | `doppler.program-bundle/v1` |
| receipt | `reference-receipt.json` | `doppler.reference-receipt/v1` |

A top-level `bundle-summary.json` (`doppler.bundle-summary/v1`) names
every emitted artifact and records per-stage status and blockers. A
failure in any stage aborts the pipeline; partial artifacts remain for
triage.

Use `--convert-config <path|json>` to run the convert stage ahead of
intake (useful for new models). Use `--skip-capture` with
`--reference-report <path> --reference-transcript <path>` to replay a
pre-existing capture (used by CI).

### Lower-level subcommands

```bash
node src/cli/doppler-cli.js program-bundle --config '{
  "manifestPath": "models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json",
  "referenceReportPath": "tests/fixtures/reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-18T13-33-38.973Z.json",
  "conversionConfigPath": "src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json",
  "outputPath": "examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json"
}'
```

`doppler intake` and `doppler reference-receipt` are the stage-level
subcommands; `doppler bundle` calls them in sequence via the shared
`performIntake` helper and the shared
`src/tooling/reference-verify.js` capture helpers.

Equivalent tool scripts:

```bash
npm run program-bundle:export -- --manifest <manifest.json> --reference-report <report.json> --out <bundle.json>
npm run program-bundle:reference -- --manifest <manifest.json> --out <bundle.json> --surface browser
npm run program-bundle:reference:gemma4-int4ple
npm run program-bundle:check
npm run program-bundle:parity
```

`program-bundle:reference` is the clean proof lane. It runs one bounded
`verify` request with a fixed prompt and token budget, captures the actual
`result.report` object returned by the selected provider, writes that report to
`reports/program-bundles/<modelId>/...reference.json`, and immediately exports
the closed bundle from that report. A manual receipt without transcript metrics
is rejected.

Program Bundle parity also works through the shared command contract:

```bash
node src/cli/doppler-cli.js verify --config '{
  "request": {
    "workload": "inference",
    "workloadType": "program-bundle",
    "programBundlePath": "examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json",
    "parityProviders": ["browser-webgpu", "node:webgpu", "node:doe-gpu"]
  }
}'
```

Default parity mode is `contract`: browser WebGPU is treated as the reference
transcript, Node/Dawn is planned for replay, and Doe.js availability is checked
without assuming the optional package is installed. Set
`programBundleParityMode: "execute"` to run the Node/WebGPU replay path.

## Failure Modes

The validator/exporter fails when:

- the manifest has no execution-v1 graph
- a reachable execution step references an undeclared kernel
- a WGSL digest does not match the checked-in digest registry
- a required artifact hash is missing
- the host contract allows dynamic imports
- a declared host entrypoint source file contains dynamic imports or DOM globals
- the reference report lacks prompt/output/token transcript identity
- the reference transcript graph hash does not match the bundle graph hash
- a Program Bundle parity request lacks a replayable bundle or asks a browser
  surface to run Node/Doe providers

## Ownership

| Area | Owner |
| --- | --- |
| model architecture, quantization, session baseline, execution graph | Doppler conversion config and manifest |
| WGSL file/entry/digest identity | Doppler execution-v1 graph |
| JS host entrypoint and allowed subset | Doppler Program Bundle contract |
| reference transcript | Doppler browser/WebGPU report |
| HostPlan, CSL, simfabric/hardware receipts | Doe |
