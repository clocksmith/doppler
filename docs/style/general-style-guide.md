# Doppler General Style Guide

Canonical source for DOPPLER's shared cross-language rules: naming, logging,
testing, and anti-patterns. Language-specific guides document only deltas.

## Invariants (Quick Reference)

Read this checklist first; open the linked sections for the behavior you change.
Mechanically checked rules still need behavioral evidence at runtime boundaries.

| Standard | Evidence |
| --- | --- |
| Find the existing owner before adding a validator, serializer, lifecycle helper or facade. Share implementations only when inputs, permissions, failure handling and ownership match. Keep one named responsibility per extraction. | [Semantic ownership](#semantic-ownership); `source:architecture:check` checks owners, dependencies and facades. Consumer behavior requires focused tests and review. |
| JSON owns policy, JavaScript orchestrates, WGSL computes. Geometry, kernel identity, dtype and materialization choices come from resolved contracts. | [Execution plane](#execution-plane-contract), [kernel selection](#kernel-selection); `source:style:check` and relevant numerical/reference tests. |
| Read manifest facts; preserve merge precedence and distinguish disabled `null` from missing `undefined`. No hidden runtime defaults or fallback. | [Configuration layers](#configuration-layers), [config guide](config-style-guide.md); affected schema, merge and command-contract tests. |
| Every acquired resource has one owner and a cleanup path on success, failure and cancellation. Share lifecycle mechanics, not different ownership policies. | [Buffer lifecycle](#buffer-lifecycle), [failure paths](javascript-style-guide.md#runtime-failure-path-invariants); focused ownership/failure tests. |
| Keep package facades thin, modules bounded, and runtime logging in the debug subsystem. | [Facade purity](#facade-purity); `source:architecture:check` and `source:style:check`. |
| Pair JavaScript with precise declarations. Add new shared implementations to strict and unused-local roots; expand existing coverage when cleaning a module. Passing a scoped gate is not whole-tree coverage. | [Language policy](#language-policy-javascript--declaration-files); `typecheck:source`, `source:unused:check`, and `exports:parity:check`. |
| Remove copied decisions before adding another validator. Generate necessary mirrors from one owner and preserve artifact/receipt identities. | [Inventory before edit](#inventory-before-edit); relevant generator check and consumer tests, not file-content imitation tests. |
| New or changed exceptions name affected paths, reason and an observable removal condition in their owning policy. Preserve existing debt; never raise a budget without inspecting the actual delta. | [Source gates](#source-style-and-architecture-gates); policy checks enforce registered limits, while removal conditions require review. |

Run checks for affected contracts under `AGENTS.md` development verification.
This checklist does not require a full suite, GPU campaign or release qualification.

## Core Principles

1. **Config as Code** - Declarative maps over imperative logic
2. **Layered Configuration** - Each layer transforms the previous
3. **Pure Functions** - Config transformations should be pure
4. **Explicit over Implicit** - No magic, document everything
5. **Single Source of Truth** - One canonical contract file, derived artifacts generated + check-synced

When the same runtime-visible metadata appears in more than one file, exactly one file must be canonical.
Remove hand-maintained copies where possible; any remaining mirror or convenience registry is generated
from that source and covered by a CI sync check, run locally only when editing either side.

Release-facing model claims follow the same rule:

- `models/catalog.json` is the source of truth for model editorial metadata, lifecycle status, and HF revision pointers.
- The external models volume is the source of truth for RDRR artifact bytes
  (manifests, shards, origin metadata). `npm run artifact-identity:check`
  cross-validates it against catalog identity.
- HF `clocksmith/rdrr` is a published subset of the catalog, filtered to approved entries.
- A model must not be surfaced as verified/release-ready unless its claim is backed by explicit smoke evidence (for example a registry verify record or committed manual-review report artifact).

Gitignored local artifacts are never a canonical CI source:

- `models/local/**` is developer-local state, not a release/CI contract surface.
- GitHub workflows must not depend on `models/local/**` being present.
- Local-only model smokes must self-skip when the artifact is absent and fail fast on manifest/shard integrity drift when it is present.

## Execution Plane Contract

Execution is partitioned into three planes:

- JSON plane: contract and policy in manifests, config assets, and rule assets.
- JS plane: orchestration and I/O with resolved config.
- WGSL plane: deterministic arithmetic kernels.

Determinism requirement:

- JSON must resolve decisions before pipeline execution.
- JS may not introduce behavior not represented in resolved config.
- WGSL may not contain runtime policy branching for selection behavior.
- Legacy compatibility is allowed only through explicit JSON rule/registry entries and must be visible in config.
- During Capsule/Runtime model execution, JS must not implement tensor arithmetic
  or tensor-layout transforms as CPU loops, typed-array math, or
  readback/compute/upload round trips.
- Required semantic geometry must be present in the resolved contract before
  allocation or dispatch; deriving it from element counts is forbidden.

The following JavaScript numeric boundaries are permitted only when their role
is explicit and inventoried by `npm run source:style:check`:

- Rig/converter artifact construction before a Capsule is signed;
- descriptor-bound artifact decoding and materialization before execution;
- application-input preprocessing before a runtime tensor is created;
- scalar sampling and other declared control-plane decisions after readback;
- observation that never feeds a modified tensor back into execution;
- deterministic CPU references in tests or quarantined reference modules; and
- numerical prototypes under `src/experimental/` while the architecture gate
  proves that no production owner imports them and support policy keeps them
  outside Doppler Run claims.

An allowed boundary may not select runtime policy, infer semantic geometry, or
serve as a fallback for a declared WGSL operation. A readback/compute/upload
round trip is always runtime tensor compute and is forbidden.

---

## Language Policy: JavaScript + Declaration Files

Doppler source code is **JavaScript** with **declaration files** (.d.ts) for every module in `src/` and `demo/`. Tests and tools may omit `.d.ts` unless they export public types.

### Why JavaScript

| Reason | Explanation |
|--------|-------------|
| **Runtime compilation** | JS/WGSL/JSON changes compile without rebuild; hot-swap plumbing is planned |
| **No generation quality difference** | No benchmarks show LLMs generate better TS than JS [1][2] |
| **Simpler toolchain** | Edit JS/WGSL/JSON and run directly; `tsc` only emits/validates `.d.ts` |

### Why .d.ts Files

| Reason | Explanation |
|--------|-------------|
| **Type context for agents** | Canonical contracts live in .d.ts; strict implementation annotations reference those contracts |
| **Consumer compatibility** | Type-aware consumers can import Doppler with full type safety via `.d.ts` |
| **Self-documenting** | Types describe interfaces without runtime cost |

### Contract Protection Layers

> Runtime validation protects untrusted boundaries; declarations protect
> consumers and refactors; tests protect behavior.

No layer substitutes for another: runtime validation fails fast on malformed
external data, and `.d.ts` declarations precisely describe consumer and
internal refactor contracts.

### File Format Summary

| Format | Use For |
|--------|---------|
| **JavaScript (.js)** | All source code; no TypeScript syntax or build migration. Checked boundary modules may import declaration types through JSDoc |
| **Declaration files (.d.ts)** | Comprehensive type specs for source modules (`src/`, `demo/`) |
| **JSON (.json)** | Static config assets, manifests, fixtures |
| **WGSL (.wgsl)** | GPU shaders |

### Evidence

| Claim | Source |
|-------|--------|
| 15% of bugs caught by types → also caught by tests | [To Type or Not to Type (ICSE 2017)][4] |
| Types help LLMs read context | Anders Hejlsberg, TS Lead Architect [3] |
| TypeScript adoption driven by validating AI output | [GitHub Octoverse 2025][5] |

### References

1. [ts-bench - TypeScript benchmark for AI agents](https://medium.com/@laiso/introducing-ts-bench-a-reproducible-benchmark-for-evaluating-ai-coding-agents-typescript-19bcf960cb7c)
2. [GitHub Octoverse - AI Feedback Loop](https://github.blog/news-insights/octoverse/typescript-python-and-the-ai-feedback-loop-changing-software-development/)
3. [Anders Hejlsberg on TypeScript and AI](https://github.blog/developer-skills/programming-languages-and-frameworks/typescripts-rise-in-the-ai-era-insights-from-lead-architect-anders-hejlsberg/)
4. [To Type or Not to Type (ICSE 2017)](https://earlbarr.com/publications/typestudy.pdf)
5. [GitHub Octoverse 2025](https://github.blog/news-insights/octoverse/octoverse-a-new-developer-joins-github-every-second-as-ai-leads-typescript-to-1/)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Model Loading                            │
├─────────────────────────────────────────────────────────────────┤
│  manifest.json → ModelConfig → Weights (GPU Buffers)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Setup                             │
├─────────────────────────────────────────────────────────────────┤
│  ModelConfig → PipelineSpec → KernelSpecs → Pipelines (cached)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Inference                                │
├─────────────────────────────────────────────────────────────────┤
│  Input tokens → [Embed → Layers → Logits] → Output tokens       │
│                    ↓                                            │
│              KernelSpecs + RuntimeUniforms → GPU Dispatch       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Layers

Schemas define config shapes and default values. Configs are instances of schemas.
Every configurable must have a schema; no runtime defaults live in JS logic.

### Config Vocabulary

- Schema: blueprint + defaults (`src/config/schema/*.schema.js`).
- Config: schema-shaped instance (default, profile, override, asset).
- Default config: `DEFAULT_*` export from a schema module.
- Profile/config asset: JSON overlay (runtime profiles, kernel paths, platforms, conversion configs).
- Override config: explicit config or programmatic overlay (highest precedence).

### Phase Injection Model

Each phase injects configs by sequential merge: existing inputs first, then default
configs, config assets, and override configs to fill gaps or override behavior.

1. Phase 1 (Conversion): model artifact + converter configs -> manifest config.
2. Phase 2 (Loading): manifest config + runtime configs + asset configs
   -> ModelConfig + pipeline/kernel specs + loaded weights.
3. Phase 3 (Inference): Phase 2 outputs + runtime inference config -> execution.

### Config Merge Order (per domain)

Not every config includes every layer (e.g., loader configs do not merge manifest
inference). Document the merge chain per domain:

- Runtime config:
  `runtimeConfig = merge(runtimeDefaultConfig, runtimeProfileConfig, runtimeOverrideConfig)`
- Manifest inference config (Phase 1 output):
  `manifestInferenceConfig = merge(manifestDefaultConfig, converterConfig.inference, artifactDerivedConfig)`
- Model inference config (Phase 2 input):
  `modelInferenceConfig = merge(manifestInferenceConfig, runtimeInferenceOverrideConfig)`
- Loader/runtime slices (loading/storage/etc):
  `loadingConfig = merge(runtimeDefaultConfig.loading, runtimeProfileConfig.loading, runtimeOverrideConfig.loading)`
- Kernel path resolution:
  `kernelPath = runtimeConfig.inference.kernelPath`
  `null` remains a valid "no explicit kernel path" result; do not invent `'auto'` or any implicit string in JS.

### No Runtime Defaults in Code

Run code should read resolved config values directly. Do not add literal fallbacks
for tunables in JS; put defaults in schemas and merge them in the config layer.

For all behavior-changing choices (kernel selection, precision mode, fallback variants),
the only fallback source is explicit config/rule assets. If policy is not present,
raise a typed configuration error instead of silently selecting an alternate behavior.

This applies to direct constructors and helper entry points too. If a test or
internal caller passes a partial object, the callee must require the resolved
field and fail with an actionable message; it must not recreate schema defaults
with `??`, destructuring defaults, or object literals in runtime code.

Allowed local defaults are limited to non-policy presentation labels, test/tool
CLI defaults, or pure constants that do not change runtime behavior. Keep those
out of inference paths and do not use them to stand in for config.

### Reuse Config Merge Utilities

Do not duplicate ad-hoc deep-merge helpers across command runners or harnesses;
browser and Node surfaces import the shared config merge utilities so behavior
is identical. Preserve explicit semantics: `undefined` means "no override",
`null` means "explicit disable/reset".

### Nullable Required Fields

For fields that can be legitimately disabled:

- `null` = explicitly disabled (valid)
- `undefined` = not specified (validation error, fail fast)

```json
{
  "attention": {
    "slidingWindow": null,
    "attnLogitSoftcapping": null
  }
}
```
Both fields are explicitly disabled (valid). Omitting either means the manifest is incomplete.

### Runtime Reads the Manifest, Never Infers

- Converter embeds all model-specific inference params in `manifest.json`
- Pipeline reads config values directly; it never detects model family or infers from architecture strings
- Run must not infer model parameters from tensor names, shapes, or heuristics
- Model-specific runtime policy must live in JSON rule assets and use exact
  manifest fields or explicit variant ID lists. Do not use substring, prefix, or
  suffix matching on model identity.

```javascript
// DON'T: infer behavior from model family strings
if (arch.includes('gemma2')) useSoftcapping = true;

// DO: read config directly
const useSoftcapping = config.attnLogitSoftcapping !== null;
```

### Kernel Selection

- Kernel selection is fully explicit in the manifest execution graph; each step pins an exact WGSL file, entry point, and content digest. No runtime resolution.
- All models must ship with an execution graph. Models without one are unsupported.
- `defaultKernelPath` does not exist in v1 manifests. The execution graph is the sole dispatch contract.
- Kernel path overrides are config-only (no harness/UI flags); selection logic lives in `src/gpu/kernels/*.js` and config files are data only.

### Manifest-Config Parity (Required Debug Check)

When a manifest-based runtime error occurs (dtype mismatch, missing field, kernel conflict):

1. **Compare the manifest to its conversion config first.** If they disagree, refresh the manifest — do not patch runtime code.
2. The conversion config (`src/config/conversion/`) is the source of truth for all manifest inference fields; the manifest is a stamped artifact.
3. Never add runtime workarounds for stale manifests. Fix the config, refresh, verify.

### Execution Graph Kernel Identity Lifecycle

- Kernel-path identity is derived from `inference.execution` in conversion config
  and stamped into the manifest execution graph.
- Runtime `kernelPath` values are inline objects generated from execution-v1.
  String registry IDs are removed and must fail during config resolution.
- Each execution step pins `kernel`, `entry`, and `kernelRef.digest`; digest drift
  is checked by the conversion/kernel digest checks.
- `src/config/kernels/registry.json` is WGSL variant metadata only, never a
  model kernel-path ID registry or runtime override lookup table.
- Add tuned kernels by updating the conversion execution graph or an explicit
  execution graph transform. Promoting the converted artifact (not committing
  the change) requires benchmark and verify evidence.
- Treat fused kernels as explicit behavior changes: preserve an equivalent
  non-fused baseline graph, and promote a fused variant only after perf (vs.
  non-fused) and correctness evidence are both green.

### Capability Transform Classification

Capability transforms are explicit policy rows in
`src/rules/inference/capability-transforms.rules.json`. Each row must declare
`kind`, `dtypeEffect`, `match`, `transforms`, `reason`, and `evidence`.

Allowed categories:

- `hardware-compatibility`: adapts to missing device features only, such as no
  `shader-f16` or no subgroups. It must not match model or platform identity.
- `runtime-session-compatibility`: adapts an already-resolved runtime session,
  such as `kvDtype=f32`, without model identity matching.
- `explicit-lane`: dispatches a requested dtype lane, such as a promoted or
  experimental f16 runtime profile. It is not a fallback.
- `platform-workaround`: disables or remaps behavior for an exact platform and
  exact model/list with evidence. It must not silently change dtype.
- `lane-mismatch-guard`: fails closed when a runtime profile requests a lane the
  selected manifest variant cannot dispatch.
- `capability-optimization`: uses a positive capability for a model-scoped
  optimization without changing dtype.
- `default-noop`: the final no-transform rule.

Dtype-changing rows expose the effect through `dtypeEffect`, and execution
receipts preserve the matched policy metadata so transformed lanes are visible.
Run `npm run capability-policy:check` after changing capability transform
policy, execution-v1 dtype lanes, or capability fallback behavior.

### Config-Only Overrides (Harness)

Runtime tunables come from runtime config only; harness/UI controls never
override them. Canonical rules: [JavaScript: Harness Override Rules](./javascript-style-guide.md#harness-override-rules)
and `config-style-guide.md`.

### Failure-Path Regression Requirement

- A fix for buffer lifecycle, readback cleanup, deferred destruction, recorder teardown, or other failure-path-only behavior includes a regression test that exercises the failing path; happy-path coverage does not catch cleanup bugs.
- A fix for command/harness parity drift includes a regression proving unsupported or mismatched contract inputs fail closed rather than being ignored or rewritten.
- Keep these tests narrow; they do not call for full-suite runs.

### Manifest-First Change Checklist

New inference knobs and model behaviors follow the [JavaScript Manifest-First Contract](./javascript-style-guide.md#manifest-first-contract).

### Tensor-Config Consistency

**Config flags must match tensor presence.** If a model has `post_feedforward_layernorm.weight` tensors, the manifest must have `postFeedforwardNorm: true`. Mismatches cause silent model failure.

**Validation**: `validateTensorConfigConsistency()` in `src/formats/rdrr/tensor-config-validator.js` checks:
- Normalization flags (`postFeedforwardNorm`, `preFeedforwardNorm`, `postAttentionNorm`)
- QK-norm flag (`queryKeyNorm`)
- Weight tying (`tieWordEmbeddings`)

**Auto-detection**: The converter derives normalization flags from tensor names to fill unspecified config fields; it must not override explicit conversion config.

**Why this matters**: In the 2026-01-19 Gemma 3 postmortem, a missing `postFeedforwardNorm: true` skipped a norm whose weights existed, and logits diverged after 26 layers.

```javascript
// DON'T: Rely on hidden family defaults for architecture-specific features
"normalization": {
  "preFeedforwardNorm": true
  // postFeedforwardNorm not set -> defaults to false -> BUG if tensors exist
}

// DO: Explicitly set all flags that match tensor presence
"normalization": {
  "preFeedforwardNorm": true,
  "postFeedforwardNorm": true  // Matches post_feedforward_layernorm.weight
}
```

**Enforcement**: Manifest validation now fails if config flags don't match tensor presence.

### Weight-Interpretation Flag Verification (Required)

**Verify weight-interpretation flags against actual weight values, never infer them from HF class names or code inspection alone.** This covers `rmsNormWeightOffset`, `scaleEmbeddings`, and any flag that changes the arithmetic applied to stored weights (e.g., `(1 + weight) * x` vs `weight * x`).

**Verification procedure** before changing any weight-interpretation flag:

1. **Sample actual weight values** from the RDRR shards (or source safetensors) for the relevant tensors.
2. **Check the distribution**:
   - `rmsNormWeightOffset: true` (Gemma-style): norm weights cluster near **0.0** (identity = `1 + 0 = 1`).
   - `rmsNormWeightOffset: false` (standard): norm weights cluster near **1.0** (identity = `1 * x`).
3. **Run inference on the affected model before AND after** the change to confirm coherence is preserved.
4. **Never patch external volume manifests** without first confirming the change produces coherent output on the patched model.

**Why this matters**: On 2026-03-18 an agent flipped Qwen 3.5's `rmsNormWeightOffset` to `false` because HF's `Qwen3_5RMSNorm` looked like standard RMSNorm. The actual norm weights (mean ~0.24–0.65) are Gemma-style `(1 + weight)`, and both Qwen models collapsed to all-newline/space output. HF class names do not indicate weight initialization.

### Storage Dtype and Materialization Dtype

Tensor storage identity and GPU materialization identity are separate contracts.

- The manifest and tensor descriptors preserve the exact stored encoding. BF16
  checkpoint bytes stay `BF16`, never declared `F16` because an older runtime
  materialized them that way.
- The resolved execution path after capability transforms determines what the
  loader must provide. Resolve its pinned shader and entry point through
  `src/config/kernels/registry.json`; no filename allowlists in loader or
  pipeline code.
- Only the selected path may request retained packed storage; an unselected
  kernel elsewhere in the manifest execution closure cannot change weight loading.
- If the selected kernel consumes another dtype, materialization is an explicit
  BF16-to-F16 or BF16-to-F32 step and the weight descriptor records the
  materialized dtype.
- A capability transform that replaces a selected kernel also changes the
  loader requirement derived from that path. Packed storage must never reach a
  kernel registered for a different weight dtype.

A change to this contract needs regression coverage of both the direct
retained-storage lane and a transformed or alternate lane that requires
materialization.

### KV Cache Dtype Policy

- KV cache dtype defaults to `f16` when supported.
- Force `f32` only when required (no `shader-f16`) or explicitly configured (e.g., `runtime.inference.kvcache.forceF32Softcap` for softcapping).
- No new defaults that silently upgrade to `f32`.

### Performance Invariants (F32 Policy)

F32 is a correctness fallback, not a performance default. When `shader-f16` is available:

- Prefer `f16` for activations, KV cache, and intermediate buffers.
- Any `f32` path must be explicitly configured (manifest or runtime) and logged once per session.
- No hidden dtype escalation via runtime fallbacks; use schema defaults and explicit switches.

Without `shader-f16`, `f32` is a required capability constraint, not an optimization.

### Runtime Profile Separation

**Conversion configs** (`src/config/conversion/`) are the sole source of truth for conversion-time inference and execution fields stamped into the manifest.

**Runtime profiles** (`src/config/runtime/profiles/` and other profile wrappers under `src/config/runtime/`) extend runtime defaults for different use cases without duplicating command or workload names. The browser harness loads them (`runtimeProfile`) and merges runtime overrides on top (see [Config Merge Order](#config-merge-order-per-domain)). Place overrides under the correct section: `runtime.shared`, `runtime.loading`, or `runtime.inference`. Discover checked-in profiles with `doppler profiles --json` instead of guessing IDs from filenames.

Manifest + runtime config feed ModelConfig → PipelineSpec → KernelSpec → Dispatch.

### Layer 1: Model Manifest (JSON)

Raw model metadata from RDRR format:

```json
{
  "hidden_size": 2048,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "intermediate_size": 8192,
  "vocab_size": 32000,
  "num_hidden_layers": 24,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000,
  "scale_embeddings": true,
  "rms_norm_weight_offset": true
}
```

### Layer 2: ModelConfig (type declarations)

With derived values:

```typescript
// model-config.d.ts
interface ModelConfig {
  // From manifest (validated)
  hiddenSize: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  numLayers: number;
  rmsNormEps: number;
  ropeTheta: number;

  // Derived
  isGQA: boolean;              // numKvHeads !== numHeads
  kvHeadDim: number;           // hiddenSize / numKvHeads

  // Feature flags (booleans)
  features: ModelFeatures;

  // Device-specific tuning
  device: DeviceConfig;
}
```

### Layer 3: PipelineSpec (type declarations)

Sequence of operations:

```typescript
// pipeline/types.d.ts
interface PipelineSpec {
  embed: { kernel: string; scale: number };
  layers: LayerSpec[];
  logits: { kernel: string; tieWeights: boolean };
}

interface LayerSpec {
  attention: {
    variant: string;
    ropeTheta: number;
    ropeLocalTheta?: number;  // Gemma dual RoPE
  };
  ffn: {
    variant: string;
    activation: 'silu' | 'gelu' | 'relu';
  };
  norms: NormSpec[];  // 2 for standard, 4 for sandwich
}
```

### Layer 4: KernelSpec (type declarations)

GPU dispatch parameters:

```typescript
// kernel-specs.d.ts
interface KernelSpec {
  pipelineKey: string;                    // Cache key
  constants: Record<string, number>;      // Baked into pipeline
  uniformSize: number;                    // Bytes
  workgroups: WorkgroupFn;                // Dispatch calculation
}

type WorkgroupFn = (uniforms: Record<string, number>) => [number, number, number];
```

### Layer 5: Runtime Uniforms (type declarations)

Values that change each inference:

```typescript
// kernel-uniforms.d.ts
interface RuntimeUniforms {
  seqLen: number;      // Tokens this pass
  startPos: number;    // Position for RoPE
  kvLen: number;       // KV cache length
  isPrefill: boolean;  // Prefill vs decode
}
```

---

## Buffer Lifecycle

### Principles

1. **Bucket by size** - Power-of-2 for small buffers (256B, 512B, 1KB, 2KB, ...)
2. **Reuse aggressively** - Return buffers to pool instead of destroying
3. **Coarse steps for large buffers** - Avoid 2x jumps on huge allocations
4. **Explicit lifecycle** - Callers acquire and release buffers; no GC
5. **Deferred destruction** - Release after GPU work completes

### DON'T: Per-Tensor Allocation

```javascript
// BAD - creates fragmentation, no reuse
const buffer = device.createBuffer({ size: exactTensorSize });
// ... use buffer ...
buffer.destroy();
```

### DO: Pool Acquisition

```javascript
// GOOD - bucketed allocation with reuse
const buffer = bufferPool.acquire(exactTensorSize);
// ... use buffer ...
bufferPool.release(buffer);
```

### Why This Matters

GPU buffer creation is expensive (~1ms). Coarse-bucket pooling amortizes it and reduces fragmentation; rounding-up waste is far cheaper than allocation overhead.

---

## JSON-Driven Layer Plans

Pipeline step ordering can be overridden via JSON without code changes.

### Default Behavior

When no pipeline plan is specified, DOPPLER uses the optimized hardcoded layer path.

### Override Mechanism

Conversion configs may define `inference.pipeline` to drive per-layer step order. Runtime config can supply `runtime.inference.pipeline` for ad-hoc experiments.

```json
{
  "runtime": {
    "inference": {
      "pipeline": {
        "steps": [
          { "op": "input_norm" },
          { "op": "attention" },
          { "op": "post_attention_norm" },
          { "op": "residual" },
          { "op": "pre_ffn_norm" },
          { "op": "ffn" },
          { "op": "post_ffn_norm" },
          { "op": "residual" }
        ]
      }
    }
  }
}
```

### When to Use

- Debugging layer ordering issues
- Testing new normalization patterns
- Comparing sandwich vs standard norm
- A/B testing fusion strategies

### When NOT to Use

Production inference: leave it unset so the optimized hardcoded path runs.

---

## Loader Layer Configuration

The loader transforms raw weight files (GGUF/SafeTensors) into GPU buffers. It has its own config concerns separate from inference.

### Quantization Format Constants

Format constants (block sizes, byte layouts) are spec invariants imported from one module; see [JavaScript: Quantization Format Constants](./javascript-style-guide.md#quantization-format-constants).

### Dtype Defaults

Default dtypes for conversion should come from config, not hardcoded strings:

```javascript
// DON'T: Hardcoded dtype strings scattered across converter
const outputDtype = options.dtype || 'f16';
const embeddingDtype = options.embeddingDtype || 'f16';

// DO: Reference schema defaults
import { getQuantizationDefaults } from '../config/schema/index.js';
const defaults = getQuantizationDefaults();
const outputDtype = options.dtype ?? defaults.defaultWeightDtype;
const embeddingDtype = options.embeddingDtype ?? defaults.defaultEmbeddingDtype;
```

### Memory Limits

Cache sizes and memory thresholds also come from config, e.g.
`getStorageDefaults().expertCache.maxSize` instead of a literal `256 * 1024 * 1024`.

---

## File Size Guidelines

| Threshold | Lines | Action |
|-----------|-------|--------|
| **Target** | 200-400 | Ideal file size |
| **Soft limit** | 750 | Consider splitting |
| **Hard limit** | 999 | Blocking ceiling for new or compliant source files |

When splitting files:
- Extract cohesive functionality into separate modules
- Group by feature, not by type (e.g., `attention.js` not `helpers.js`)
- Keep related code together; don't split just to hit a number

The ceiling is not a target. A file that owns multiple state machines,
execution planes, workload families, or lifecycle boundaries must split even
when it remains below 999 lines.

Existing over-limit files are governed in
`tools/policies/source-architecture-policy.json`; each entry records the
file's current maximum and named extraction boundaries. An inherited file may
shrink but not grow; once it reaches the limit, its entry must be removed.
The architecture check compares this ledger with `DOPPLER_POLICY_BASE_REF`
(`HEAD` for an uncommitted local change). Adding a soft-limit review or raising
`reviewedLines` fails unless the changed review records a `debtAuthorization`
SHA-256 digest exactly matching the externally supplied
`DOPPLER_ARCHITECTURE_DEBT_AUTHORIZATION`. Lowering or removing debt never
needs authorization. Every other policy relaxation (global limit increases,
new legacy ceilings, cross-owner dependency additions, exceptions, facades,
experimental bridges, standalone reachability exemptions, weaker
constitutional graphs) needs a fresh top-level `debtAuthorization` matching
the same externally administered CI variable.

## Semantic Ownership

- Treat top-level directories as subsystem boundaries, not file buckets.
- Shared behavior belongs to the narrowest feature or subsystem that owns its
  semantics.
- Cross-subsystem reuse passes through a neutral contract module rather than a
  broad orchestration file.
- Do not create new catch-all `utils`, `helpers`, `common`, or `shared` modules.
  A narrowly named feature utility is acceptable when the feature owns the
  complete invariant.
- Existing generic utility modules are compatibility debt, not precedent.
- Policy, resolved plans, resource ownership, execution, and observation must
  remain distinct responsibilities.

The canonical ownership and dependency policy lives in
`tools/policies/source-architecture-policy.json`; prose must not maintain a
second changing owner inventory. Its `genericModuleReviews` ledger classifies
every inherited generic filename as a narrow semantic owner, a pure
compatibility facade, or bounded legacy debt with named extraction targets.

Every production module has exactly one declared subsystem owner. The policy
graph applies to every owner and fails on cycles, forbidden dependency
direction, production-to-experimental imports, unreachable production modules,
stale exceptions, or ambiguous ownership. Compatibility modules are owned
facades, never a generic legacy subsystem.

## Facade Purity

Public entrypoints and compatibility facades may:

- re-export stable contracts;
- validate and normalize external inputs;
- delegate to a feature owner;
- translate errors into the public envelope.

They may not own workload algorithms, mutable runtime policy, backend-specific
selection, resource state machines, or independent execution semantics. A
facade retained for compatibility must have a declared consumer and remain
behaviorally exercised.

## Source Style and Architecture Gates

Static gates; run them when a change touches what they check (source modules,
imports, facades, declarations, numeric boundaries, or their policy files):

```bash
npm run source:style:check
npm run source:architecture:check
npm run source:architecture:check -- --base <git-revision>
```

The style gate permits implementation type annotations only in the explicit
strict roots in `tsconfig.source-strict.json`. These import canonical contracts
from sibling declarations; API descriptions remain in declarations. Strict roots
enable `noImplicitAny` and `strictNullChecks` and include negative-use tests.
`tools/policies/source-type-debt.json` inventories unchecked implementations and
existing declaration `any` use; additions fail the type gate. The style gate also
enforces mechanically detectable execution-plane invariants: every typed-array
numeric candidate is classified in the source-compute policy; production runtime
tensor compute, readback/compute/upload paths, stale or missing classifications,
runtime geometry inference, raw runtime logging outside allowed entrypoints,
unowned policy/default construction, and imprecise sibling declarations fail.
`npm run source:style:sync` preserves annotations in checked strict roots.

The architecture gate rejects unknown or ambiguous source owners, dependency
cycles, forbidden dependency edges for every production owner,
production-to-experimental edges, unreachable production modules,
implementation declarations in declared facades, unreviewed soft- or
hard-limit files, stale dependency exceptions, and stale generic module
ownership entries. Temporary exceptions are bounded debt, never
repository-wide conformance.

### Published Package Closure Gate

`src/.npmignore` is generated by `npm run package:closure:sync`; never edit it
by hand. `npm run public:boundaries:check` verifies that every packed JavaScript,
declaration, and resource file is reachable from a public package entry point
and rejects repository-only files. Package count and size ceilings ratchet:
raise one only after auditing the generated closure and confirming the delta is
an intentional public-runtime dependency, then set the smallest practical new
ceiling rather than bypassing closure validation.

---

## File Organization

The maintained `src/` tree is in [AGENTS.md § Directory Structure](../../AGENTS.md#directory-structure);
module ownership is canonical in `tools/policies/source-architecture-policy.json`
(see [Semantic Ownership](#semantic-ownership)). Do not maintain a second tree here.

---

## Naming Conventions

### Files

| Pattern | Example | Use For |
|---------|---------|---------|
| `kebab-case.js` | `model-config.js` | JavaScript modules |
| `snake_case.wgsl` | `matmul_f16.wgsl` | WGSL shaders |

### Type Declarations (.d.ts)

```typescript
// types.d.ts
// PascalCase for types/interfaces
interface ModelConfig { }
interface KernelSpec { }
type GPUVendor = 'apple' | 'nvidia';

// Suffix with purpose
interface MatmulUniforms { }     // Uniform struct
interface MatmulConstants { }   // Pipeline constants
type MatmulVariant = string;    // Enum-like
```

### Variables

```javascript
// camelCase for variables and functions
const modelConfig = parseModelConfig(manifest);
const kernelSpec = KERNEL_SPECS.attention(modelConfig);

function selectMatmulVariant(ctx) { }
```

### Constants

```javascript
// UPPER_SNAKE_CASE for module-level constants
const WORKGROUP_SIZES = { ... };
const TILE_SIZES = { ... };
const FUSION_THRESHOLDS = { ... };

// camelCase for derived/local constants
const workgroupSize = WORKGROUP_SIZES[vendor].matmul;
```

### WGSL

`UPPER_SNAKE_CASE` constants, `snake_case` struct fields and functions; see
[WGSL Style Guide: Naming Conventions](./wgsl-style-guide.md#naming-conventions).

---

## Error Handling

### Validation Errors (Fail Fast)

```javascript
function parseModelConfig(manifest) {
  // Validate required fields
  if (!manifest.hidden_size) {
    throw new Error('manifest.hidden_size is required');
  }
  if (manifest.hidden_size % manifest.num_attention_heads !== 0) {
    throw new Error(
      `hidden_size (${manifest.hidden_size}) must be divisible by ` +
      `num_attention_heads (${manifest.num_attention_heads})`
    );
  }
  // ...
}
```

### Runtime Errors (Descriptive)

```javascript
async function dispatch(kernel, ...args) {
  const spec = this.specs[kernel];
  if (!spec) {
    throw new Error(
      `Unknown kernel: "${kernel}". ` +
      `Available: ${Object.keys(this.specs).join(', ')}`
    );
  }
  // ...
}
```

### GPU Errors (Device Loss)

```javascript
async function runKernel(...args) {
  const device = getDevice();
  if (device.lost) {
    throw new Error('GPU device lost. Call initDevice() to recover.');
  }
  // ...
}
```

---

## Logging

Use the unified debug module (`src/debug/index.js`: `log.*` levels, `trace.*`
categories), never raw `console.*`, in runtime code. Raw console output is
allowed only in `tools/` entry points, `tests/` harnesses, demo entry points,
benchmark result tables, and one-time startup messages in `src/gpu/device.js`.
API details: [JavaScript Style Guide: Logging](./javascript-style-guide.md#logging).

### No Ad-Hoc Debug Logging

**Do not add temporary log statements to source files for debugging.** All debugging must use:

1. **Existing trace categories** via config assets or custom config
2. **Config-driven probes** for inspecting specific values
3. **Extended trace categories** if needed visibility doesn't exist (permanent addition, not temporary)

This supports the hot-swap architecture: change configs, not code.

```javascript
// DON'T: Add temporary debug logging
log.info('Config', `LayerTypes: ${layerTypes.slice(0, 10).join(', ')}`);

// DO: Use config-driven tracing (via runtimeConfig URL param)
const traceConfig = {
  shared: { debug: { trace: { enabled: true, categories: ['all'] } } }
};

// DO: Use probes for specific values
const probeConfig = {
  shared: { debug: { probes: [{ id: 'layer0', stage: 'layer_out', layers: [0] }] } }
};
```

---

## Testing

Which tests to run follows the repo `AGENTS.md` Development verification section. Test behavior;
don't add tests that only assert one file restates another or pin config literals.

### Unit Tests (Kernel Correctness)

```javascript
// tests/kernels/browser/test-page.js (excerpt)
const gpu = await testHarness.getGPU();
const output = await testHarness.runMatmul(gpu.device, A, B, 4, 4, 8);
if (!output) throw new Error('matmul output missing');
```

### Config Checks (Rule Coverage)

```javascript
// tests/kernels/browser/test-page.js (excerpt)
const ctx = { bDtype: 'q4k', M: 1, hasSubgroups: true, N: 2048 };
const variant = selectMatmulVariant(ctx);
if (variant !== 'q4_fused') throw new Error(`expected q4_fused, got ${variant}`);
```

### Integration Runs (E2E)

On demand; not a commit, push, or deploy gate.

```javascript
// In browser console (demo diagnostics or harness)
await runBrowserSuite({ workload: 'inference', modelId: 'gemma-3-270m-it-q4k-ehf16-af32' });
```

---

## Enforcement

- The kernel harness (`tests/kernels/browser/test-page.js`) validates kernel registry and WGSL overrides; run it when changing those.
- Preserve working configs as checked-in config assets (configuration is documentation).

### Inventory Before Edit

When a failure reveals a repeated drift class, inventory the whole class (grep, or that class's existing check) before editing individual files. Classify the full set, then batch fixes by decision type: fix to runtime, fix to declarations, remove stale surface, or quarantine as pending-feature/debt with owner and expiry.

Prefer removing the duplication that causes the drift. Add a `--check` tool only when the drift cannot be removed and keeps recurring (existing examples: `npm run kernels:digests:check`, `npm run exports:parity:check`, `npm run pending:check`), and run such checks only when a change edits what they check.

---

## Documentation

### Code Comments

Comment why, not what, and only for non-obvious behavior; see
[JavaScript: Comments](./javascript-style-guide.md#comments).

### Config Documentation

```javascript
/* Matmul variant selection rules.
 *
 * Rules are evaluated in order - first match wins.
 *
 * Variants:
 * - q4_fused_*: Direct Q4K dequant + matmul (fastest for Q4K)
 * - gemv_*: M=1 optimized (decode path)
 * - f16*: F16 compute (when supported)
 * - f32: Fallback
 */
const MATMUL_VARIANTS = [
  // ...
];
```

---

## Anti-Patterns

### DON'T: Mix Configuration Layers

```javascript
// BAD - kernel dispatch knows about manifest
async function runAttention(manifest, ...args) {
  const numHeads = manifest.num_attention_heads;  // Wrong layer!
}

// GOOD - kernel dispatch uses KernelSpec
async function runAttention(spec, uniforms, ...args) {
  const workgroups = spec.workgroups(uniforms);
}
```

### DON'T: Duplicate Configuration

```javascript
// BAD - same threshold in multiple places
if (N > 4096) { /* in matmul.js */ }
if (N > 4096) { /* in gemv.js */ }

// GOOD - single source of truth
const THRESHOLDS = { multicol: { minN: 4096 } };
if (N > THRESHOLDS.multicol.minN) { ... }
```

### DON'T: Implicit Defaults

See [No Runtime Defaults in Code](#no-runtime-defaults-in-code) and
[JavaScript: Use Bare Fallback Literals](./javascript-style-guide.md#dont-use-bare-fallback-literals).

---

## See Also

- [WGSL Style Guide](./wgsl-style-guide.md) - Shader conventions
- [JavaScript Style Guide](./javascript-style-guide.md) - Kernel wrapper conventions
- [Config Style Guide](./config-style-guide.md) - Merge order and config category rules
