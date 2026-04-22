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

Doppler should define a model once and make that definition portable:

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

## Bundle Contents

A Program Bundle must declare:

- `programContractVersion`
- `dopplerExecutionGraphVersion`
- `webgpuSubset`
- `wgslSubset`
- `jsSubset`
- `unsupportedFeaturePolicy`
- model id and source package identity
- manifest path and manifest hash
- execution-v1 graph and graph hash
- declared JS host entrypoint or program factory
- declared WGSL modules with file, entry point, and digest identity
- artifact roles, paths, sizes, and hashes
- tokenizer and input/prompt contract
- runtime profile for reference execution
- capture profile for provider/capture runs
- provider requirements such as WebGPU feature subset
- bounded reference transcript, including token IDs, stop reason, and logits
  hashes when required by the validation lane

The bundle must be closed: no hidden imports, no implicit kernels, no
runtime-discovered shader strings, and no undeclared model behavior.
Unsupported features must reject early when `unsupportedFeaturePolicy` is
`fail`; there is no downgrade from portable program to best-effort capture.

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

## JavaScript Boundary

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

## Kernel Identity

Execution-v1 is the bridge. A Program Bundle must not depend on:

- string `runtime.inference.kernelPath` IDs
- legacy `kernelPlan`
- removed kernel-path registry assets
- implicit defaults that change execution behavior

Runtime `kernelPath` may be `null` or an inline object generated from
execution-v1. `null` means an explicit no-override/reset value and must not be
rewritten into defaults.

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
