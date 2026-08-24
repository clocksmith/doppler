# Glimmer Architectural Generalization Campaign

Status: `lowered-parity-investigation`

Owner: Doppler Model Release Foundry

Portfolio role: subordinate execution campaign, not an independently ranked
strategy or product vector

## Purpose

Muse Glimmer 30B tests whether Doppler can turn an independently designed,
heterogeneous model into an architecture-specialized JavaScript product without
adding model-family behavior to Runtime. It is one Glimmer product with
independently qualified capability profiles, not three composable products.

The initial market lane is installed workstation software: Node applications,
Electron applications, and VS Code extensions. Browser support is a separately
qualified target. Bun remains experimental under the existing Bun product
qualification contract and cannot inherit Node evidence.

## Current evidence boundary

The canonical source-truth receipt is
[`reports/model-ir-v2/glimmer-30b.model-ir-receipt.json`](../../reports/model-ir-v2/glimmer-30b.model-ir-receipt.json).
It pins `meta-models/Muse-Glimmer-30B` revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68` and represents three components:

- text decoder;
- vision encoder;
- vision projector.

The source receipt remains architecture evidence and records both
`text.generate` and `vision.encode` as `unlowered`. The later lowerability and
semantic-lowering receipts are separate artifacts: they now admit and bind a
Pack-scoped `text.generate` execution candidate while preserving the source
topology. `vision.encode` remains unlowered.

The deterministic
[`text.generate` lowerability audit](../../reports/model-ir-v2/glimmer-30b.lowerability-audit.json)
tests the source ModelIR against the generic heterogeneous-text vocabulary
without inspecting the model name. It now reports `lowerable=true`: local and
full attention resolve to admitted v2 lowerings and no required state kind is
unimplemented.

The semantic lowering receipt binds those generic mechanisms into an explicit
conversion/session/execution contract for `muse-glimmer-30b-text-f16-af32` and
retains seven rejected candidates beside the accepted lowering candidate. The
generic execution engine has physical AMD/RADV evidence for that candidate and
for a BF16-storage sibling, but none is promotable: all retained reports fail
source-token parity. The strongest retained BF16 report matches source tokens
through generation index 6 and diverges at index 7. Boundary-comparison receipts
are diagnostic only and explicitly record `promotionEligible=false`.

The operational meanings are not inferred from config names. The source
receipt pins the Hugging Face Transformers implementation and source hashes,
records exact symbol and line spans, and admits zero unresolved text operational
facts. In particular, `output_multiplier` is represented on the final logits
path, not as an attention multiplier. This supports a lowered execution
candidate; it does not establish source parity, qualification, or product
support.

DFlash is absent from the current ModelIR and requires a separate pinned source,
revision, license, tensor, and reference-behavior intake. The current receipt has
one generic `text.kv-state`; it does not yet encode 39 local ring-buffer states
and 13 global states. Split-KV memory estimates are design inputs, not measured
allocation evidence.

## Product and Pack boundary

The product identity is `muse-glimmer-30b`. The intended capability profiles are:

| Profile | Capability | Current status |
| --- | --- | --- |
| `text-core` | Text generation with the fixed local/local/local/global schedule | Lowered candidate; source parity failing |
| `text-dflash` | Text generation with an independently admitted speculative drafter | DFlash not represented |
| `multimodal` | Text plus perception encoder and projector | Vision entry point unlowered |

Pack v2 currently defines a self-contained program and does not define Pack
dependencies or extension composition. Each promoted capability profile must
therefore produce a complete, independently signed and qualified Pack. No profile
may load undeclared artifacts from another Pack.

Each Pack may contain multiple pre-qualified TargetPlans. Initial target order is
Node WebGPU, Electron WebGPU, browser WebGPU, and then explicit Doe-native targets.
Runtime selection must fail closed when the requested target is absent. Doe is an
optional native executor, never a browser requirement or hidden fallback.

## Qualification stages

### 1. Text source parity

- Lower `text.generate` from the pinned ModelIR without a Glimmer-named Runtime branch.
- Bind tokenizer, chat template, weights, output head, sampling policy, and source revision.
- Pass deterministic greedy logit and token parity against a pinned reference executor.
- Qualify one self-contained `text-core` Pack on Node WebGPU before making application claims.
- Embed the exact Pack in one Node or Electron application and retain task-level outcomes.

### 2. Architecture specialization

- Encode the 39 local and 13 global layer schedule directly in ModelIR and TargetPlans.
- Replace generic `text.kv-state` with fixed local ring and growing global state contracts.
- Qualify architecture-specific attention, normalization, gating, output, prefill, and decode paths.
- Search quantization and physical layouts through Forge while retaining rejected candidates.
- Compare output quality, load, memory, prefill, decode, and complete application latency against the unspecialized baseline.

### 3. DFlash speculative execution

- Admit the exact DFlash source repository, revision, license, weights, and reference behavior separately.
- Represent the drafter, hidden-state taps, verification graph, and acceptance policy in ModelIR and the Pack.
- Treat a block size of 16 as including its anchor, allowing up to 15 proposed tokens.
- Keep target features, proposals, verification, and acceptance on the GPU unless a qualified contract requires otherwise.
- Freeze any promotion threshold, including a proposed `1.3x` end-to-end gain, before candidate measurement.
- Reproduce incumbent figures independently; upstream results are baselines, not Doppler evidence.
- Keep `text-core` recommended when DFlash fails parity, reliability, memory, or the frozen gain threshold.

### 4. Multimodal and Doe targets

- Lower and qualify `vision.encode`, projector execution, and text/vision integration.
- Preserve text-only operation without requiring perception artifacts.
- Qualify explicit standard WebGPU and Doe-native TargetPlans against identical model semantics and task oracles.
- Record Doe provider, backend, native-library, artifact, and hardware identity; Doe is zero-daemon, not zero-native-code.
- Promote Bun only through its independent three-workload product qualification, never by inheriting Node or Electron results.

## Promotion evidence

The campaign earns product evidence only when all claims cite immutable artifacts
for the exact capability profile and target. Required evidence includes source and
license identity, ModelIR and TargetPlan hashes, Pack semantic root, reference
parity, physical hardware identity, cold and warm load, memory, prefill, ordinary
decode, speculative decode where applicable, failures, recovery, and complete
application-task outcomes.

The decisive external gate is one independently operated Node, Electron, or VS
Code application selecting Doppler specifically for Glimmer and completing a
predeclared workflow against the strongest relevant incumbent. Tokens per second
alone cannot satisfy that gate.

## Non-goals

- Creating another Ouroboros strategy, company, or commercial product vector.
- Claiming that model-artifact licensing covers every runtime dependency.
- Treating Meta's quality or speed reports as Doppler results.
- Advertising three-Pack composition before a dependency contract exists.
- Promoting Bun, browser, multimodal, DFlash, or Doe support from source intake alone.
