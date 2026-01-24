# DOPPLER JavaScript Style Guide

JavaScript conventions for kernel wrappers and pipeline code.

See [General Style Guide](./GENERAL_STYLE_GUIDE.md#language-policy-javascript--declaration-files) for the rationale behind JavaScript-first development.

## Core Principle: Config as Code

Replace conditional logic with declarative configuration maps.

```
Model Manifest → ModelConfig → PipelineSpec → KernelSpec → Execution
     ↓              ↓              ↓              ↓              ↓
 manifest.json   .d.ts types    Op sequence   GPU params    Dispatch
```

## JSON Rule Maps (Required for Selection Logic)

Any selection of kernel variants, dtype strings, or op names must use JSON rule maps.

- Store rules under `src/rules/<domain>/.../*.rules.json`.
- Use `selectRuleValue()` from `src/rules/rule-registry.js` (or `src/gpu/kernels/rule-registry.js` for kernel-only call sites).
- Avoid inline ternaries/if-else for choosing variant strings or dtypes.
- Keep rule maps consistent and reusable: `match` + `value` keys only, no ad-hoc fields.

Example:

```javascript
// Good: rule map drives selection
const variant = selectRuleValue('kernels', 'matmul', 'phase', { isDecode });

// Avoid: inline selection logic for variants/dtypes
const variant = isDecode ? 'decode' : 'prefill';
```

## Manifest-First Contract

Any new inference knob must be wired end-to-end:
- Add to `ManifestInferenceSchema` (and converter defaults if needed)
- Populate in converter mapping (preset + HF config)
- Merge in `src/config/merge.js` with `_sources`
- Validate in `parseModelConfigFromManifest()`
- Add tests that assert manifest values and runtime overrides

Do not reintroduce runtime model detection or preset fallbacks.

## Nullable Required Fields

For nullable-but-required fields:
- `null` = explicitly disabled (valid)
- `undefined` = missing (validation error)

Validation should check `=== undefined` for nullable fields and `== null` for non-nullable fields.

## Kernel Path Only

Kernel selection overrides must use `kernelPath`. `kernelPlan` is removed and must not be reintroduced.
Kernel path overrides are config-only; harness/UI surfaces must not set kernel selection via ad-hoc flags.

## Harness Override Rules

- Harness options must live in config (`runtime.shared.harness`).
- Per-field URL overrides are forbidden; use `runtimePreset`, `runtimeConfig`, `runtimeConfigUrl`, or `configChain`.
- Harness/UI controls must not override runtime tunables (prompt, max tokens, sampling, trace/log levels, warmup/timed runs).
See `CONFIG_STYLE_GUIDE.md` for merge order and category rules.

## Runtime Configuration (Performance Invariants)

Runtime code must respect dtype and performance invariants from config and device capabilities.

- Do not hardcode `f32` fallbacks when `shader-f16` is available.
- If a `f32` path is required, require an explicit config flag and log once per session.
- Treat capability limits (no `shader-f16`) as constraints, not “optimizations.”

If you need to switch dtype or kernel variants, put the decision in rule maps or schema-driven config, not in ad-hoc conditionals.

---

## Types: .d.ts Files

All type definitions live in `.d.ts` files. JavaScript files contain no type annotations.

| File | Contains |
|------|----------|
| `module.js` | Clean code, no JSDoc types |
| `module.d.ts` | All type definitions for that module |

Agents read `.d.ts` files directly for type context. No need to duplicate types in JS.

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

**No JSDoc in JS files.** Descriptions, parameter docs, and examples belong in `.d.ts` files.

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

Kernels that support both immediate (`run*`) and batched (`record*`) execution MUST keep both implementations in sync.

```javascript
// DON'T: Divergent implementations
export async function runGather(...args) {
  const workgroups = Math.ceil(numElements / (WORKGROUP_SIZES.VEC4_THREADS * 4));
}
export async function recordGather(...args) {
  const workgroups = Math.ceil(numElements / 256);  // Different constant!
}

// DO: Share the calculation
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
2. Are both using the same constants?
3. Are both producing identical GPU dispatches?

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

Inline rule arrays are not allowed in production code. Put all selection rules
in JSON and evaluate them via the rule registry. The only exception is tests
that validate the matcher itself.

### JSON Rule Maps (Required for Selection Logic)

Selection rules must live in JSON and be evaluated by the rule registry. This
keeps selection logic data-only, auditable, and consistent across kernels and
inference code.

**Required pattern:**

1. Define rules in `src/**/rules/*.rules.json`
2. Use `selectRuleValue()` from `src/rules/rule-registry.js`
3. Pass a context object with all decision inputs (no inline ternaries/ifs)

```json
// src/gpu/kernels/rules/softmax.rules.json
{
  "variant": [
    { "match": { "hasSubgroups": true, "isSmall": true }, "value": "small_subgroup" },
    { "match": { "hasSubgroups": true }, "value": "subgroup" },
    { "match": { "isSmall": true }, "value": "small" },
    { "match": {}, "value": "default" }
  ]
}
```

```javascript
import { selectRuleValue } from '../rules/rule-registry.js';

function selectSoftmaxVariant(innerSize) {
  const caps = getKernelCapabilities();
  const isSmall = innerSize <= getKernelThresholds().softmax.smallThreshold;
  return selectRuleValue('softmax', 'variant', {
    hasSubgroups: caps.hasSubgroups,
    isSmall,
  });
}
```

**Exception:** Only trivial, non-selection UI toggles may use inline conditionals.

### Rule Matcher Utility (Internal)

```javascript
// utils/rule-matcher.js
// Types in rule-matcher.d.ts

export function matchesRule(rule, context) {
  for (const [key, expected] of Object.entries(rule)) {
    const actual = context[key];

    if (typeof expected === 'object' && expected !== null && !Array.isArray(expected)) {
      // Range check: { gt: 100, lt: 1000 }
      if ('gt' in expected && actual <= expected.gt) return false;
      if ('lt' in expected && actual >= expected.lt) return false;
      if ('eq' in expected && actual !== expected.eq) return false;
    } else {
      // Exact match
      if (actual !== expected) return false;
    }
  }
  return true;
}

export function selectByRules(rules, context, defaultValue) {
  const rule = rules.find(r => matchesRule(r.match, context));
  return rule?.value ?? defaultValue;
}
```

Do not call `matchesRule` or `selectByRules` directly in production code. Use
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

Note: `src/gpu/kernel-selector.js` is a thin re-export for backward compatibility; selection lives in `src/gpu/kernels/*`.

### Rule Registry

Use `selectRuleValue()` from `src/rules/rule-registry.js` (or the kernel-local
registry) for all kernel variant selection. Keep rules as pure data and build
the context object before matching.

### Registry Derivation

`src/gpu/kernels/kernel-configs.js` derives from `src/config/kernels/registry.json`. Do not add parallel kernel registries.

### Selection Keys

Each kernel selects variants based on context:
- **Dimensions**: M, N, K, seqLen, hiddenSize
- **Dtypes**: weight dtype (f16, q4k, q6k), activation dtype
- **Capabilities**: hasF16, hasSubgroups, device limits
- **Platform**: vendor-specific tuning (Apple, AMD, NVIDIA)

### Pipeline Caching

Pipelines are cached per variant/context combination in `pipeline-cache.js`:

```javascript
// Pipeline key includes all variant-affecting params
const key = `matmul_${variant}_${M}_${N}_${K}_${aDtype}_${bDtype}`;
let pipeline = pipelineCache.get(key);
if (!pipeline) {
  pipeline = await createPipeline(variant, constants);
  pipelineCache.set(key, pipeline);
}
```

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

Centralize quantization block sizes in a single module. Format-specific constants are **invariants** derived from the quantization spec—they should never be redefined.

```javascript
// loader/quantization-constants.js
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

```javascript
// gpu/kernel-executor.js

export class KernelExecutor {
  constructor(model, device) {
    this.device = device;
    this.pipelines = new Map();
    // Pre-compute all specs from model config (pure transformation)
    this.specs = Object.fromEntries(
      Object.entries(KERNEL_SPECS).map(([name, factory]) => [name, factory(model)])
    );
  }

  async dispatch(kernel, buffers, uniforms) {
    const spec = this.specs[kernel];
    if (!spec) throw new Error(`Unknown kernel: ${kernel}`);

    const pipeline = await this.getPipeline(spec);
    const uniformBuffer = this.createUniformBuffer(spec.uniformSize, uniforms);
    const [x, y, z] = spec.workgroups(uniforms);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.createBindGroup(pipeline, uniformBuffer, buffers));
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async getPipeline(spec) {
    let pipeline = this.pipelines.get(spec.pipelineKey);
    if (!pipeline) {
      pipeline = await createPipelineWithConstants(spec.pipelineKey, 'main', spec.constants);
      this.pipelines.set(spec.pipelineKey, pipeline);
    }
    return pipeline;
  }
}
```

---

## Naming Conventions

### Files

```
gpu/kernels/
  matmul.js              # Main kernel wrapper
  matmul-utils.js        # Shared utilities (if needed)
  fused-matmul-norm.js   # Fused variant

config/
  model-config.js        # ModelConfig and parser
  kernel-specs.js        # KernelSpec factories
  config-tables.js       # WORKGROUP_SIZES, TILE_SIZES, etc.

types/
  gpu.d.ts               # GPU type declarations (optional)
  inference.d.ts         # Inference type declarations (optional)
```

### Type Declarations (.d.ts)

```typescript
// types.d.ts
interface ModelConfig { }
type GPUVendor = 'apple' | 'nvidia' | 'amd' | 'intel' | 'default';
```

### Constants

```javascript
// UPPER_SNAKE_CASE for config tables
const WORKGROUP_SIZES = { ... };
const TILE_SIZES = { ... };
const FUSION_THRESHOLDS = { ... };

// camelCase for derived values
const workgroupSize = WORKGROUP_SIZES[vendor].matmul;
```

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

### DON'T: Scatter Constants

```javascript
// BAD - same value in multiple places
function selectMatmul() {
  if (N > 4096) { ... }  // Magic number
}
function selectGemv() {
  if (N > 4096) { ... }  // Same magic number, different file
}

// GOOD - centralized table
const THRESHOLDS = {
  gemvMulticol: { minN: 4096 },
};
```

### DON'T: Mix Config Layers

```javascript
// BAD - uniform creation knows about manifest
function createUniforms(manifest) {
  view.setUint32(0, manifest.hidden_size);  // Wrong layer!
}

// GOOD - uniforms only know about runtime values
function createUniforms(uniforms) {
  view.setUint32(0, uniforms.seqLen);
}
```

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
const maxTokens = opts.maxTokens ?? getRuntimeConfig().inference.batching.maxTokens;
```

**Rule:** If a value has a fallback, the fallback must be a named constant or config getter. If the value is truly required, don't provide a fallback—fail fast with a descriptive error.

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

Why: The first debugging question is "where did this value come from?" Source
tracking answers that instantly.

---

## Logging

All library code MUST use the unified debug module (`debug/index.js`) instead of raw `console.*` calls. Exceptions: `tests/` harnesses, demo entry points, and one-time startup messages in `src/gpu/device.js`.

### Import

```javascript
import { log, trace } from '../debug/index.js';
```

### Log Levels

Use the appropriate log level based on message importance:

```javascript
log.error('Module', 'Critical failure');        // Always shown (except silent)
log.warn('Module', 'Recoverable issue');        // Shown at warn+ level
log.info('Module', 'Normal operation');         // Shown at info+ level (default)
log.verbose('Module', 'Detailed info');         // Shown at verbose+ level
log.debug('Module', 'Implementation detail');   // Shown at debug level only
```

### Trace Categories

For kernel/pipeline debugging, use trace categories:

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
| `inference/pipeline/layer.js` | `'Layer'` or `'Pipeline'` |
| `loader/doppler-loader.js` | `'Loader'` |
| `demo/app.js` | `'App'` or `'DopplerDemo'` |

### Exceptions

Raw `console.*` is acceptable in:

1. **Demo entry points** (`demo/`, `tests/harness.html`) - User-facing status output
2. **Test files** (`tests/kernels/`, `tests/benchmarks/`, `tests/training/`) - Test harness output
3. **Benchmarks** - Formatted results tables
4. **One-time startup** - GPU device info in `device.js`

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

Tests are JavaScript. Same rules as source code: clean JS, types in `.d.ts` if needed.

### File Naming

```
tests/kernels/browser/
  test-page.js               # Kernel harness

tests/benchmarks/
  README.md                  # Benchmark entrypoint notes

tests/training/browser/
  test-page.js               # Training harness
```

### Structure

```javascript
// In browser console (tests/harness.html, mode: kernels)
const gpu = await window.testHarness.getGPU();
const result = await window.testHarness.runMatmul(gpu.device, a, b, m, n, k);
console.log('max error', Math.max(...result));
```

### What to Test

| Test Type | Purpose | Location |
|-----------|---------|----------|
| **Kernel harness** | GPU output matches CPU reference | `tests/kernels/browser/test-page.js` |
| **Training harness** | Training kernel validation | `tests/training/browser/test-page.js` |
| **Benchmark** | Performance regression detection | `src/inference/browser-harness.js` |

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

### Test Before Migrate

**Gate:** Tests must exist for a module before that module's source is migrated to JS.

---

## See Also

- [WGSL Style Guide](./WGSL_STYLE_GUIDE.md) - Shader conventions
- [General Style Guide](./GENERAL_STYLE_GUIDE.md) - General patterns
