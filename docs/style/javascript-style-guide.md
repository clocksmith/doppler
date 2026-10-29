# Doppler JavaScript Style Guide

JavaScript conventions for kernel wrappers and pipeline code. Shared naming,
logging, testing, and anti-pattern policy, and the
[JavaScript-first rationale](./general-style-guide.md#language-policy-javascript--declaration-files),
are canonical in the [General Style Guide](./general-style-guide.md).

## Invariants (Quick Reference)

Rules that cause bugs when violated; each has a fuller section below.

- **Role Boundaries** — JSON owns policy; JS owns orchestration; WGSL owns math. See [Role Boundaries](#role-boundaries-json-js-wgsl).
- **JSON Rule Maps** — any selection of kernel variants/dtype/op must use `selectRuleValue()` on a rule map. No inline ternaries. See [JSON Rule Maps](#json-rule-maps-required-for-selection-logic).
- **Manifest-First Contract** — new inference knobs must land in `ManifestInferenceSchema`, the converter, `src/config/merge.js` with `_sources`, `parseModelConfigFromManifest()`, and a narrow parse/merge test. No runtime family detection, no hidden fallbacks. See [Manifest-First Contract](#manifest-first-contract).
- **Nullable Required Fields** — `null` = explicitly disabled (valid); `undefined` = missing (validation error). Check `=== undefined` for nullable, `== null` otherwise. See [Nullable Required Fields](#nullable-required-fields).
- **Kernel Path Only** — kernel overrides use `kernelPath`. `kernelPlan` is removed and must not return. Overrides are config-only. See [Kernel Path Only](#kernel-path-only).
- **Runtime Performance Invariants** — no hardcoded `f32` fallbacks when `shader-f16` is available; if `f32` is required, gate on explicit config and log once. Dtype decisions live in rule maps, not conditionals. See [Runtime Configuration (Performance Invariants)](#runtime-configuration-performance-invariants).
- **Failure-Path Ownership** — every acquired resource has one owner and one deterministic cleanup path. `createBuffer()`, pooled acquires, `mapAsync()`, staging readback all release on every throw. Use `try/finally` or shared helpers. See [Runtime Failure-Path Invariants](#runtime-failure-path-invariants).
- **Harness Override Rules** — runtime tunables are config-only. Harness URLs accept only `runtimeProfile`, `runtimeConfig`, `runtimeConfigUrl`, `configChain`. See [Harness Override Rules](#harness-override-rules).
- **Command/Session/Observation Boundaries** — `request.intent` is the sole active command authority; resolved sessions are immutable numeric policy; observation context controls evidence only. See [Inference Contract Boundaries](#inference-contract-boundaries).
- **Semantic Plan Purity** — operation plans are deterministic, immutable, and JSON-safe; GPU objects enter only at binding. See [Inference Contract Boundaries](#inference-contract-boundaries).
- **Adapter Parity** — browser, Node, immediate, recorded, benchmark, and replay adapters consume one normalized semantic request or plan. See [Adapter Parity](#adapter-parity).

## Unused imports and locals

Remove unused bindings when editing a module. Preserve imports required for
module initialization as explicit side-effect imports. Check re-exports,
declaration consumers, computed loading, and declared standalone entrypoints
before deleting an exported symbol or module.

`npm run source:unused:check` uses TypeScript's unused-local diagnostics on the
JavaScript implementations listed in `tsconfig.source-unused.json`; add cleaned
modules there to expand coverage. The gate also runs with `source:style:check`.
It does not establish general type correctness or check unused parameters;
the existing strict-source and declaration checks remain separate.

## Core Principle: Config as Code

Replace conditional logic with declarative configuration maps.

```
Model Manifest → ModelConfig → PipelineSpec → KernelSpec → Execution
     ↓              ↓              ↓              ↓              ↓
 manifest.json   .d.ts types    Op sequence   GPU params    Dispatch
```

## Inference Contract Boundaries

These stable invariant IDs are normative:

| ID | Invariant | Mechanical enforcement |
|---|---|---|
| `INV-CMD-001` | `request.intent` is the sole active-run intent authority. Profiles declare `compatibleIntents`; runtime config never authors or mirrors active intent. | command-surface and runtime-profile schema tests |
| `INV-SESSION-002` | Numeric inference consumes one immutable resolved runtime session constructed after manifest, execution-v1, capability, kernel-path, and dtype resolution. | resolved-session tests |
| `INV-ATTN-003` | Attention code must not read module-global runtime config. | `npm run inference:boundaries:check` |
| `INV-PLAN-004` | A semantic operation plan is immutable, deterministic, JSON-safe, and contains no GPU buffers, recorders, closures, or cleanup callbacks. | attention-plan tests |
| `INV-PARITY-005` | Browser, Node, immediate, recorded, benchmark, and replay adapters consume the same normalized semantic request or plan. Executors may differ only in transport, submission, retention, completion tasks, and evidence capture. | plan parity tests + `npm run source:architecture:check` |
| `INV-RESOURCE-006` | Every concrete resource has tagged ownership: `borrowed`, `scopeOwned`, `submitOwned`, `transferred`, or `retained`. Ownership changes use explicit scope operations. | resource-scope failure and alias tests |
| `INV-RECEIPT-007` | Structural refactors preserve canonical behavior receipts containing command, session, plan, operation order, dtype transitions, resource events, and first failure boundary. | refactor-receipt tests |
| `INV-COMPUTE-008` | During Capsule/Runtime model execution, JavaScript never performs tensor arithmetic or tensor-layout transforms; declared WGSL programs own those operations. Reviewed Rig construction, artifact codecs/materialization, host input preprocessing, scalar control, observation, and reference boundaries cannot act as runtime fallbacks. | `npm run source:style:check` + kernel reference tests |
| `INV-GEOMETRY-009` | Runtime-required tensor and spatial geometry is explicit in ModelIR, TargetPlan, manifest, or resolved session data and is never inferred from counts or identity heuristics. | `npm run source:style:check` + manifest/pipeline contract tests |

The contract layering is:

```text
CommandContext + ResolvedRuntimeSession + ObservationContext
                         ↓
               Semantic Operation Plan
                         ↓ bind resources
                 Bound Operation Plan
                         ↓
                      Executor
```

Intent may govern permitted observation and evidence requirements. It must not
change dtype, kernel, fusion, geometry, layout, or numeric inference behavior.
Run a listed enforcement check when a change touches that invariant.

## Role Boundaries (JSON, JS, WGSL)

- JSON owns policy and selection state (`manifest`, runtime profiles/config, rule maps).
- JS owns orchestration: validation, pipeline setup, resource lifecycle, dispatch, and readback.
- WGSL owns only math and memory transforms.

JavaScript may compute scalar dispatch sizes, byte offsets, validation bounds,
and host-side sampling explicitly assigned to the control plane. The other
reviewed numeric roles (Rig construction, byte decoding/materialization, input
preprocessing, observation, references, `src/experimental/` prototypes) and
their limits are canonical in the
[General Execution Plane Contract](./general-style-guide.md#execution-plane-contract).
The source-compute policy inventories these roles by exact module and symbol;
unreviewed or stale entries fail the style gate.

JavaScript may not read a tensor back to implement attention, normalization,
activation, projection, pooling, rearrangement, scheduling, expert combination,
or another tensor operation before uploading or returning the computed model
result. Those operations require a registered WGSL program and a deterministic
reference test.

Geometry is policy-bearing model semantics. Grid height, grid width, patch
ordering, merge factors, head geometry, and layout strides must be explicit in
the resolved contract. A pipeline must fail before allocation or dispatch when
required geometry is absent or inconsistent; `Math.sqrt(elementCount)` and
shape reconstruction from tensor names are prohibited.

For deterministic behavior, variant decisions come from `selectRuleValue()`
([JSON Rule Maps](#json-rule-maps-required-for-selection-logic)); missing rules
or required config throw a configuration error; compatibility aliases come only
from explicit registry/JSON entries.

## JSON Rule Maps (Required for Selection Logic)

Any selection of kernel variants, dtype strings, or op names must use JSON rule maps.

- Store rules under `src/rules/<domain>/.../*.rules.json`.
- Use `selectRuleValue()` from `src/rules/rule-registry.js` (or `src/gpu/kernels/rule-registry.js` for kernel-only call sites).
- Avoid inline ternaries/if-else for choosing variant strings or dtypes.
- Keep rule maps consistent and reusable: `match` + `value` keys only, no ad-hoc fields.
- Do not duplicate the same rule-backed condition in multiple JS modules. Put
  shared normalization/validation in one helper and keep the selection condition
  in the JSON rule map.

Example:

```javascript
// Good: rule map drives selection
const variant = selectRuleValue('kernels', 'matmul', 'phase', { isDecode });

// Avoid: inline selection logic for variants/dtypes
const variant = isDecode ? 'decode' : 'prefill';
```

## Manifest-First Contract

Any new inference knob or model behavior must be wired end-to-end:
- Add to `ManifestInferenceSchema` (and converter defaults/fixtures if needed)
- Populate in converter mapping (explicit conversion config + HF config)
- Merge in `src/config/merge.js` with `_sources` tracking
- Validate in `parseModelConfigFromManifest()` (null vs undefined rules)
- Cover parse/merge behavior (manifest value, override precedence) with a narrow unit test; don't pin config literals

Do not reintroduce runtime model detection or hidden family fallbacks.
If a field affects inference behavior, absence is a configuration error at the
runtime boundary. Do not use `??` or destructuring defaults to fill manifest or
runtime tunables in pipeline, cache, execution-plan, or kernel-dispatch code.

## Nullable Required Fields

Semantics (`null` = disabled, `undefined` = missing) are canonical in
[General: Nullable Required Fields](./general-style-guide.md#nullable-required-fields).
Validate nullable fields with `=== undefined` and non-nullable fields with `== null`.

## Kernel Path Only

Kernel selection overrides must use `kernelPath`. `kernelPlan` is removed and must not be reintroduced.
Runtime `kernelPath` values must be inline execution-v1-derived objects or `null`; string registry IDs are removed.
Kernel path overrides are config-only; harness/UI surfaces must not set kernel selection via ad-hoc flags.

## Harness Override Rules

- Runtime tunables come from runtime config only. Harness/UI controls must not override them (prompt selection, max tokens, sampling temperature/topK/topP, trace categories, log levels, warmup/timed runs).
- Command intent and harness selection stay in explicit command/suite context, never in `runtime.shared.*`.
- Per-field URL overrides are forbidden; use `runtimeProfile`, `runtimeConfig`, `runtimeConfigUrl`, or `configChain`.
- To tweak a tunable, create a runtime profile or pass `--config` with a runtime config file.
See `config-style-guide.md` for merge order and category rules.

## Shared Contract + Engine Overlay (Bench Tools)

Cross-engine benchmark tools must build and pass two explicit config objects:

1. Shared benchmark contract (prompt/template, sampling tuple, seed, token budgets, run counts, cache mode).
2. Engine overlay (engine-specific execution knobs).

Rules:
- Parse/validate shared workload values once, then reuse for all engines.
- Keep Doppler-only knobs (`batchSize`, `readbackInterval`, `kernelPath`) and vendor backend/session flags out of the shared contract.
- Build runtime payloads from `(sharedContract, engineOverlay)` helper functions instead of duplicating JSON assembly in each call site.

## Runtime Configuration (Performance Invariants)

Run code must respect dtype and performance invariants from config and device capabilities.

- Do not hardcode `f32` fallbacks when `shader-f16` is available.
- If a `f32` path is required, require an explicit config flag and log once per session.
- Treat capability limits (no `shader-f16`) as constraints, not “optimizations.”
- Execution-v1 capability transforms must consume the classified JSON policy
  row (`kind`, `dtypeEffect`, `evidence`) from the resolver. Do not add inline
  dtype fallback branches in pipeline, cache, or kernel wrapper code.
- Lane receipts must preserve declared lane, executed lane, applied transforms,
  and the matched capability policy when capability transforms are resolved.

Dtype or kernel-variant switches belong in rule maps or schema-driven config, never ad-hoc conditionals.

## Runtime Failure-Path Invariants

Runtime resource ownership must stay deterministic on both success and failure paths.

- Any acquired runtime resource has one clear owner and one deterministic cleanup path.
- Any `createBuffer()`, pooled-buffer acquire, `mapAsync()`, staging readback allocation, or temporary recorder allocation must release, destroy, or unmap on every throw path.
- Prefer `try/finally` or a shared helper that owns the cleanup boundary; do not duplicate cleanup sequences inline across call sites.
- GPU readback uses shared helpers, not ad hoc `createBuffer() + copyBufferToBuffer() + mapAsync()` sequences. A documented exception states why the helper is insufficient and keeps its cleanup path local, explicit, and test-covered.

---

## Types: .d.ts Files

Canonical type definitions live in `.d.ts` files, which agents read directly for
type context. JavaScript remains executable without compilation. Modules listed
in `tsconfig.source-strict.json` may use JSDoc type imports and local
annotations to check their implementation against those definitions; do not
duplicate public interfaces in JavaScript.

```javascript
// module.js - clean, no type annotations
function writeUniforms(view, uniforms) {
  view.setUint32(0, uniforms.seqLen, true);
  view.setUint32(4, uniforms.startPos, true);
}
```

```typescript
// module.d.ts - comprehensive type specs
interface KernelUniforms {
  seqLen: number;
  startPos: number;
  kvLen: number;
}

declare function writeUniforms(view: DataView, uniforms: KernelUniforms): void;
```

### Comments

Descriptions and examples belong in `.d.ts` files. Implementation JSDoc is
limited to type-checking annotations in the declared strict roots.

When a change touches declarations or strict-root annotations,
`npm run source:style:check` verifies the governed `src/` and `demo/` trees. Use
`npm run source:style:sync` only after sibling declaration coverage is complete.

**Inline comments are rare.** Code should be self-explanatory. If you need a comment, first consider renaming or refactoring.

#### When to Comment

```javascript
// DO: Workaround for external bug
// Chrome 119: readback fails without fence
device.queue.onSubmittedWorkDone();

// DO: Why something counterintuitive is correct
// Intentionally no await - fire and forget
device.queue.submit([encoder.finish()]);
```

#### When NOT to Comment

```javascript
// DON'T: Describe what code does (it's obvious)
// Increment the counter
counter++;

// DON'T: Repeat the function name
// This function writes uniforms
function writeUniforms(view, u) { ... }

// DON'T: Explain language features
// Use async/await for promises
const result = await fetch(url);

// DON'T: Add JSDoc descriptions (use .d.ts)
/**
 * Writes uniform values to a DataView.  <-- This belongs in .d.ts
 */
function writeUniforms(view, u) { ... }
```

#### Section Headers

Use sparingly for long files. Keep them minimal:

```javascript
// === Dispatch ===

// === Pipeline Cache ===
```

---

## Kernel Wrapper Structure

```javascript
// gpu/kernels/kernel-name.js
// Types in kernel-name.d.ts

import { getDevice } from '../device.js';
import { createPipelineWithConstants, createUniformBuffer } from './utils.js';

// ═══════════════════════════════════════════════════════════════════
// UNIFORM BUFFER (must match WGSL struct)
// ═══════════════════════════════════════════════════════════════════

const UNIFORM_LAYOUT = {
  seqLen: { offset: 0, size: 4 },
  startPos: { offset: 4, size: 4 },
  kvLen: { offset: 8, size: 4 },
  _pad: { offset: 12, size: 4 },
};

const UNIFORM_SIZE = 16;

function writeUniforms(view, u) {
  view.setUint32(UNIFORM_LAYOUT.seqLen.offset, u.seqLen, true);
  view.setUint32(UNIFORM_LAYOUT.startPos.offset, u.startPos, true);
  view.setUint32(UNIFORM_LAYOUT.kvLen.offset, u.kvLen, true);
  view.setUint32(UNIFORM_LAYOUT._pad.offset, 0, true);
}

// ═══════════════════════════════════════════════════════════════════
// PIPELINE CACHE
// ═══════════════════════════════════════════════════════════════════

const pipelineCache = new Map();

function getPipelineKey(c) {
  return `kernel_${c.HIDDEN_SIZE}_${c.HEAD_DIM}_${c.NUM_HEADS}`;
}

async function getPipeline(constants) {
  const key = getPipelineKey(constants);
  let pipeline = pipelineCache.get(key);
  if (!pipeline) {
    pipeline = await createPipelineWithConstants('kernel_name', 'main', {
      HIDDEN_SIZE: constants.HIDDEN_SIZE,
      HEAD_DIM: constants.HEAD_DIM,
      NUM_HEADS: constants.NUM_HEADS,
      WORKGROUP_SIZE: constants.WORKGROUP_SIZE,
    });
    pipelineCache.set(key, pipeline);
  }
  return pipeline;
}

// ═══════════════════════════════════════════════════════════════════
// DISPATCH
// ═══════════════════════════════════════════════════════════════════

export async function runKernel(input, weights, output, constants, uniforms) {
  const device = getDevice();
  const pipeline = await getPipeline(constants);

  const uniformBuffer = createUniformBuffer(UNIFORM_SIZE, (view) => {
    writeUniforms(view, uniforms);
  });

  // Workgroups = ceil(seqLen * NUM_HEADS / WORKGROUP_SIZE)
  const totalThreads = uniforms.seqLen * constants.NUM_HEADS;
  const workgroups = Math.ceil(totalThreads / constants.WORKGROUP_SIZE);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, createBindGroup(pipeline, uniformBuffer, input, weights, output));
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
}
```

```typescript
// gpu/kernels/kernel-name.d.ts

export interface KernelUniforms {
  seqLen: number;
  startPos: number;
  kvLen: number;
}

export interface KernelConstants {
  HIDDEN_SIZE: number;
  HEAD_DIM: number;
  NUM_HEADS: number;
  WORKGROUP_SIZE: number;
}

export function runKernel(
  input: GPUBuffer,
  weights: GPUBuffer,
  output: GPUBuffer,
  constants: KernelConstants,
  uniforms: KernelUniforms
): Promise<void>;
```

---

## run/record Function Parity

Kernels that support both immediate (`run*`) and recorded (`record*`) execution
must consume one semantic plan and one interpreter ([Adapter Parity](#adapter-parity)).

```javascript
// DON'T: Divergent implementations
export async function runGather(...args) {
  const workgroups = Math.ceil(numElements / (WORKGROUP_SIZES.VEC4_THREADS * 4));
}
export async function recordGather(...args) {
  const workgroups = Math.ceil(numElements / 256);  // Different constant!
}

// Transitional minimum: share the calculation
const GATHER_ELEMENTS_PER_WG = WORKGROUP_SIZES.VEC4_THREADS * 4;  // 64 × 4 = 256

export async function runGather(...) {
  const workgroups = Math.ceil(numElements / GATHER_ELEMENTS_PER_WG);
}
export async function recordGather(...) {
  const workgroups = Math.ceil(numElements / GATHER_ELEMENTS_PER_WG);
}
```

**Checklist when editing `run*` functions:**

1. Does a corresponding `record*` function exist?
2. Do both consume the same immutable semantic plan?
3. Is every numeric/kernel/fusion/layout choice made before executor selection?
4. Are ownership differences expressed by immediate/recorded resource scopes?
5. Is any new semantic decision covered by the normalized-plan parity test?

## Adapter Parity

The run/record rule applies to every execution adapter, not only kernels:

- normalize command, session, workload, and operation policy before selecting
  browser, Node, immediate, recorded, benchmark, or replay execution;
- bind runtime resources only after the semantic request or plan is complete;
- adapters differ only in transport, submission, resource retention,
  completion, readback scheduling, and evidence capture; no adapter-local
  dtype, kernel, fusion, geometry, layout, fallback, or workload decisions;
- preserve the canonical behavior receipt across structural refactors
  (resolved policy, operation order, ownership events, output identity, first
  failure boundary).

Semantic choices that differ between adapters belong in the resolved contract
and stay visible in the receipt.

---

## Config Maps Over If/Else

### DON'T: Decision Trees

```javascript
// BAD - 70 lines of nested if/else, hard to audit
function selectMatmulVariant(M, N, K, aDtype, bDtype, transposeB, caps) {
  if (bDtype === 'q4k') {
    if (M === 1) {
      if (caps.hasSubgroups) {
        if (N > 4096) {
          return 'q4_fused_multicol';
        }
        return 'q4_fused';
      }
      return 'q4_fused_batched';
    }
    // ... 50 more lines
  }
  // ...
}
```

### Avoid: Rule Arrays in JS

Inline rule arrays are not allowed in production code (tests of the matcher
itself excepted). Selection rules live in JSON and are evaluated via the rule
registry; see [JSON Rule Maps](#json-rule-maps-required-for-selection-logic).

### Rule Matcher Utility (Internal)

`src/rules/rule-matcher.js` implements `matchesRule()` (exact values or
operator objects such as `{ gt: 100, lt: 1000 }`) and `selectByRules()`, which
throws when no rule matches. Read the source for the operator set. Do not call
`matchesRule` or `selectByRules` directly in production code. Use
`selectRuleValue()` so all selection passes through the rule registry.

---

## Kernel Selection Architecture

Kernel variant selection is **per-op and context-driven**, with pipelines cached per variant.
Selection logic lives in `src/gpu/kernels/*.js`. Config files provide data only; they do not execute selection.

### Structure

```
src/gpu/kernels/
  matmul.js              # Matmul selection + dispatch
  attention.js           # Attention selection + dispatch
  rmsnorm.js             # RMSNorm selection + dispatch
  ...
  pipeline-cache.js      # Shared pipeline caching
  utils.js               # Shared utilities, KERNEL_CONFIGS
  index.js               # Barrel export
```

`src/gpu/kernel-selector.js` is only a thin backward-compatibility re-export.

### Rule Registry

All kernel variant selection uses `selectRuleValue()` ([JSON Rule Maps](#json-rule-maps-required-for-selection-logic)).
Keep rules as pure data and build the context object before matching.

### Registry Derivation

`src/gpu/kernels/kernel-configs.js` derives from `src/config/kernels/registry.json`. Do not add parallel kernel registries.

### Selection Keys

Each kernel selects variants based on context:
- **Dimensions**: M, N, K, seqLen, hiddenSize
- **Dtypes**: weight dtype (f16, q4k, q6k), activation dtype
- **Capabilities**: hasF16, hasSubgroups, device limits
- **Platform**: vendor-specific tuning (Apple, AMD, NVIDIA)

### Pipeline Caching

Pipelines are cached per variant/context combination in `pipeline-cache.js`; the
cache key includes every variant-affecting param (variant, dimensions, dtypes).
See `getPipeline()` in [Kernel Wrapper Structure](#kernel-wrapper-structure).

### Threshold Wiring Pattern

Pull thresholds into the context object before matching so rules stay data-only:

```javascript
import { getKernelThresholds } from '../../config/schema/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';

function selectSoftmaxVariant(innerSize) {
  const { smallThreshold } = getKernelThresholds().softmax;
  const ctx = { innerSize, isSmall: innerSize <= smallThreshold };
  return selectRuleValue('softmax', 'variant', ctx);
}
```

### DON'T: Inline Selection at Call Sites

```javascript
// BAD - selection logic scattered across pipeline code
if (weights.dtype === 'q4k' && seqLen === 1) {
  await runQ4KFused(...);
} else {
  await runMatmulF32(...);
}
```

### DO: Selection in Kernel Module

```javascript
// GOOD - selection encapsulated in kernel module
import { runMatmul } from './kernels/matmul.js';
await runMatmul(input, weights, output, { M, N, K, caps });
// matmul.js handles variant selection internally
```

---

## Config Tables

Centralize magic numbers into typed tables:

```javascript
// gpu/kernels/config-tables.js
// Types in config-tables.d.ts

/** Optimal workgroup sizes by vendor and kernel */
export const WORKGROUP_SIZES = {
  apple: { matmul: 128, attention: 32, ffn: 128, norm: 256, rope: 256 },
  nvidia: { matmul: 256, attention: 64, ffn: 256, norm: 256, rope: 256 },
  amd: { matmul: 64, attention: 64, ffn: 64, norm: 128, rope: 128 },
  intel: { matmul: 128, attention: 32, ffn: 128, norm: 128, rope: 128 },
  default: { matmul: 256, attention: 64, ffn: 256, norm: 256, rope: 256 },
};

/** Tile sizes by operation and precision */
export const TILE_SIZES = {
  matmul: { f32: 16, f16: 16, q4k: 32 },
  attention: { f32: 32, f16: 64 },
  ffn: { f32: 16, f16: 16 },
  norm: {},
  rope: {},
};

/** Fusion decision thresholds */
export const FUSION_THRESHOLDS = {
  matmul_rmsnorm: { maxM: 1, maxN: 8192 },
  ffn_swiglu: { maxSeqLen: 1, maxIntermediate: 16384 },
  attention_flash: { minSeqLen: 128 },
};

/** Feature flag → kernel variant mapping */
export const FEATURE_VARIANTS = {
  attention: [
    { features: ['gqa', 'causal'], variant: 'attention_gqa_causal' },
    { features: ['gqa'], variant: 'attention_gqa' },
    { features: ['causal'], variant: 'attention_causal' },
    { features: [], variant: 'attention_standard' },
  ],
  rope: [
    { features: ['dualRoPE'], variant: 'rope_dual' },
    { features: [], variant: 'rope_standard' },
  ],
};
```

### Vec4 Workgroup Pattern

When a kernel processes 4 elements per thread (vec4), document the relationship:

```javascript
// gpu/kernels/constants.js
export const WORKGROUP_SIZES = {
  DEFAULT: 256,           // Standard scalar kernels
  VEC4_THREADS: 64,       // Vec4 kernels: 64 threads × 4 elements = 256 elements
};

// DON'T: Bare literal in vec4 dispatch
const workgroups = Math.ceil(numElements / 256);  // Where does 256 come from?

// DO: Express the relationship explicitly
const ELEMENTS_PER_WG = WORKGROUP_SIZES.VEC4_THREADS * 4;  // 64 × 4 = 256
const workgroups = Math.ceil(numElements / ELEMENTS_PER_WG);
```

This pattern applies to: `gather.js`, `residual.js`, and any kernel using `vec4<f16>` or `vec4<f32>` loads.

---

## Quantization Format Constants

Quantization block sizes are **invariants** derived from the format spec, defined once in a single module and never redefined.

```javascript
// config/schema/quantization.schema.js
export { QK_K, K_SCALE_SIZE } from '../converter/quantizer.js';

// Block byte sizes - derived from format spec, never hardcode elsewhere
export const Q4K_BLOCK_BYTES = 144;   // Q4_K: 2 + 2 + K_SCALE_SIZE + QK_K/2
export const Q6K_BLOCK_BYTES = 210;   // Q6_K: QK_K/2 + QK_K/4 + QK_K/16 + QK_K
export const Q8_0_BLOCK_BYTES = 34;   // Q8_0: 2 + QK_K (scale + quants)
export const Q8_0_BLOCK_SIZE = 32;    // Elements per Q8_0 block
```

### DON'T: Redefine Format Constants

```javascript
// BAD - redefined in multiple files, easy to drift
// doppler-loader.js:
const Q4K_K = 256;
const Q4K_BLOCK_BYTES = 144;

// dequant.js:
const Q6K_BLOCK_BYTES = 210;  // Same constant, different file

// Bare magic number in calculation:
const numBlocks = buffer.byteLength / 144;  // What is 144?

// GOOD - import from single source
import { Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES } from './quantization-constants.js';
const numBlocks = buffer.byteLength / Q4K_BLOCK_BYTES;
```

---

## Config Flow Layers

### Layer 1: Manifest → ModelConfig

```javascript
// config/model-config.js
// Types in model-config.d.ts

export function parseModelConfig(manifest, device) {
  const numHeads = manifest.num_attention_heads;
  const numKvHeads = manifest.num_key_value_heads ?? numHeads;

  return {
    hiddenSize: manifest.hidden_size,
    numHeads,
    headDim: manifest.head_dim ?? manifest.hidden_size / numHeads,
    numKvHeads,
    intermediateSize: manifest.intermediate_size,
    vocabSize: manifest.vocab_size,
    numLayers: manifest.num_hidden_layers,

    // Derived
    kvHeadDim: manifest.hidden_size / numKvHeads,
    isGQA: numKvHeads !== numHeads,

    // Features
    features: {
      scaleEmbeddings: manifest.scale_embeddings ?? false,
      rmsNormWeightOffset: manifest.rms_norm_weight_offset ?? false,
      sandwichNorm: !!manifest.pre_feedforward_layernorm,
      dualRoPE: !!manifest.rope_local_base_freq,
    },

    // Device
    workgroupSize: WORKGROUP_SIZES[device.vendor]?.matmul ?? WORKGROUP_SIZES.default.matmul,
  };
}
```

### Layer 2: ModelConfig → KernelSpecs

```javascript
// config/kernel-specs.js
// Types in kernel-specs.d.ts

/** Kernel spec factories - pure functions from model config */
export const KERNEL_SPECS = {
  attention: (model) => ({
    pipelineKey: `attn_h${model.numHeads}_d${model.headDim}_kv${model.numKvHeads}`,
    constants: {
      NUM_HEADS: model.numHeads,
      HEAD_DIM: model.headDim,
      NUM_KV_HEADS: model.numKvHeads,
      WORKGROUP_SIZE: model.workgroupSize,
    },
    uniformSize: 16,
    workgroups: (u) => [model.numHeads, Math.ceil(u.seqLen / 32), 1],
  }),

  ffn: (model) => ({
    pipelineKey: `ffn_h${model.hiddenSize}_i${model.intermediateSize}`,
    constants: {
      HIDDEN_SIZE: model.hiddenSize,
      INTERMEDIATE_SIZE: model.intermediateSize,
      WORKGROUP_SIZE: model.workgroupSize,
    },
    uniformSize: 8,
    workgroups: (u) => [Math.ceil(u.seqLen * model.intermediateSize / model.workgroupSize), 1, 1],
  }),

  rmsnorm: (model) => ({
    pipelineKey: `norm_h${model.hiddenSize}_o${model.features.rmsNormWeightOffset ? 1 : 0}`,
    constants: {
      HIDDEN_SIZE: model.hiddenSize,
      WEIGHT_OFFSET: model.features.rmsNormWeightOffset ? 1 : 0,
      WORKGROUP_SIZE: model.workgroupSize,
    },
    uniformSize: 4,
    workgroups: (u) => [u.seqLen, 1, 1],
  }),
};
```

### Layer 3: KernelSpecs → Executor

The executor precomputes every spec from model config once (a pure
transformation), throws on an unknown kernel name, caches pipelines by
`spec.pipelineKey`, and dispatches `spec.workgroups(uniforms)` with the same
encoder sequence as [Kernel Wrapper Structure](#kernel-wrapper-structure).

---

## Naming Conventions

File (`kebab-case.js`, e.g. `matmul.js`, `fused-matmul-norm.js`), type,
variable, and constant naming: [General: Naming Conventions](./general-style-guide.md#naming-conventions).

---

## Anti-Patterns

### DON'T: Create Pipelines Per-Dispatch

```javascript
// BAD - pipeline creation is expensive (~10ms)
async function runKernel() {
  const pipeline = await device.createComputePipelineAsync(...);  // Every call!
  // ...
}

// GOOD - cache and reuse
const pipelineCache = new Map<string, GPUComputePipeline>();
async function runKernel() {
  const pipeline = pipelineCache.get(key) ?? await createPipeline(...);
}
```

### DON'T: Hardcode Uniform Sizes

```javascript
// BAD - magic number, easy to mismatch with WGSL
const uniformBuffer = device.createBuffer({ size: 32 });

// GOOD - derive from layout
const UNIFORM_LAYOUT = {
  seqLen: { offset: 0, size: 4 },
  startPos: { offset: 4, size: 4 },
  // ...
};
const UNIFORM_SIZE = Object.values(UNIFORM_LAYOUT).reduce((sum, f) => sum + f.size, 0);
```

Scattered constants and mixed config layers: see [General: Anti-Patterns](./general-style-guide.md#anti-patterns).

### DON'T: Use Bare Fallback Literals

```javascript
// BAD - fallback hides the source of truth
const wgSize = options.workgroupSize || 256;
const maxSeq = config.maxSeqLen || 4096;
const maxTokens = opts.maxTokens || 1024;

// GOOD - reference named constant
const wgSize = options.workgroupSize ?? WORKGROUP_SIZES.DEFAULT;

// GOOD - required values should fail fast, not silently default
const maxSeq = config.maxSeqLen;
if (maxSeq === undefined) throw new Error('maxSeqLen is required');

// GOOD - use config schema defaults
const maxTokens = opts.maxTokens ?? getRuntimeConfig().inference.generation.maxTokens;
```

**Rule:** If a value has a fallback, the fallback must be a named constant or config getter. If the value is truly required, don't provide a fallback. Fail fast with a descriptive error.

---

## Debugging

### Config Source Tracking

For merged configs, track where each value came from:

```typescript
// config/merge.d.ts
interface MergedConfig {
  inference: InferenceConfig;
  _sources: Map<string, 'manifest' | 'runtime'>;
}

// Debug output: "slidingWindow: 4096 (manifest)"
```

Source tracking answers the first debugging question: where did this value come from?

---

## Logging

All library code MUST use the unified debug module (`debug/index.js`) instead of
raw `console.*`; the allowed exceptions are listed in [General: Logging](./general-style-guide.md#logging).

### Import

```javascript
import { log, trace } from '../debug/index.js';
```

### Log Levels

```javascript
log.error('Module', 'Critical failure');        // Always shown (except silent)
log.warn('Module', 'Recoverable issue');        // Shown at warn+ level
log.info('Module', 'Normal operation');         // Shown at info+ level (default)
log.verbose('Module', 'Detailed info');         // Shown at verbose+ level
log.debug('Module', 'Implementation detail');   // Shown at debug level only
```

### Trace Categories

```javascript
trace.loader('Shard 0 from OPFS');              // Model loading
trace.kernels(`matmul M=${M} N=${N} K=${K}`);   // Kernel execution
trace.attn(layerIdx, 'Using chunked decode');   // Attention (layer-aware)
trace.ffn(layerIdx, 'SwiGLU activation');       // FFN (layer-aware)
trace.logits('Top-5 tokens computed');          // Logit computation
trace.sample('Greedy: selected token 42');      // Token sampling
trace.kv(layerIdx, 'Cache updated');            // KV cache ops
trace.buffers('Pool: 128MB allocated');         // Buffer stats (expensive!)
trace.perf('Layer 0: 12.5ms');                  // Performance timing
```

### DON'T: Raw Console Calls

```javascript
// BAD - bypasses log level control, no history, inconsistent format
console.log('[Pipeline] Model loaded');
console.warn('[Attention] Fallback to CPU');
console.log(`[Matmul] M=${M}, N=${N}`);

// GOOD - respects log level, captured in history, consistent format
log.info('Pipeline', 'Model loaded');
log.warn('Attention', 'Fallback to CPU');
trace.kernels(`Matmul M=${M}, N=${N}`);
```

### Module Naming

Use consistent module names matching the file/class:

| File | Module Name |
|------|-------------|
| `gpu/kernels/matmul.js` | `'Matmul'` |
| `gpu/kernels/attention.js` | `'Attention'` |
| `inference/pipelines/text/layer.js` | `'Layer'` or `'Pipeline'` |
| `loader/doppler-loader.js` | `'Loader'` |
| `demo/demo.js` | `'DopplerDemo'` |

### Browser Console API

The debug module exposes `window.DOPPLER` for runtime debugging:

```javascript
// In browser console
DOPPLER.setLogLevel('debug');
DOPPLER.setTrace('kernels,attn');
DOPPLER.printLogSummary(20);
DOPPLER.getDebugSnapshot();
```

---

## Tests

Tests are JavaScript under the same rules as source (types in `.d.ts` if
needed). Which tests to write and run: [General: Testing](./general-style-guide.md#testing).

### File Naming

```
tests/kernels/browser/
  test-page.js               # Kernel harness

tests/benchmarks/
  README.md                  # Benchmark entrypoint notes

tests/training/browser/
  test-page.js               # Training harness
```

### What to Test

| Test Type | Purpose | Location |
|-----------|---------|----------|
| **Kernel harness** | GPU output matches CPU reference | `tests/kernels/browser/test-page.js` |
| **Training harness** | Training kernel validation | `tests/training/browser/test-page.js` |
| **Benchmark** | Performance regression detection | `src/tooling/browser-harness.js` |

### Reference Implementations

Kernel tests compare GPU output against CPU reference implementations:

```javascript
// tests/kernels/reference/matmul.js
export function matmulReference(a, b, m, n, k) {
  const c = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let l = 0; l < k; l++) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }
  return c;
}
```

---

## See Also

- [WGSL Style Guide](./wgsl-style-guide.md) - Shader conventions
- [General Style Guide](./general-style-guide.md) - General patterns
