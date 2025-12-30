# DOPPLER TypeScript Style Guide

TypeScript conventions for kernel wrappers and pipeline code.

## Core Principle: Config as Code

Replace conditional logic with declarative configuration maps.

```
Model Manifest → ModelConfig → PipelineSpec → KernelSpec → Execution
     ↓              ↓              ↓              ↓              ↓
 manifest.json   Typed config   Op sequence   GPU params    Dispatch
```

---

## Kernel Wrapper Structure

```typescript
// gpu/kernels/kernel-name.ts

import { getDevice } from '../device.js';
import { createPipelineWithConstants, createUniformBuffer } from './utils.js';

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

/** Uniform struct - MUST match WGSL exactly */
interface KernelUniforms {
  seqLen: number;
  startPos: number;
  kvLen: number;
}

/** Constants baked into pipeline - from model config */
interface KernelConstants {
  HIDDEN_SIZE: number;
  HEAD_DIM: number;
  NUM_HEADS: number;
  WORKGROUP_SIZE: number;
}

// ═══════════════════════════════════════════════════════════════════
// UNIFORM BUFFER (must match WGSL struct)
// ═══════════════════════════════════════════════════════════════════

const UNIFORM_LAYOUT = {
  seqLen: { offset: 0, size: 4 },
  startPos: { offset: 4, size: 4 },
  kvLen: { offset: 8, size: 4 },
  _pad: { offset: 12, size: 4 },
} as const;

const UNIFORM_SIZE = 16;  // Sum of all fields

function writeUniforms(view: DataView, u: KernelUniforms): void {
  view.setUint32(UNIFORM_LAYOUT.seqLen.offset, u.seqLen, true);
  view.setUint32(UNIFORM_LAYOUT.startPos.offset, u.startPos, true);
  view.setUint32(UNIFORM_LAYOUT.kvLen.offset, u.kvLen, true);
  view.setUint32(UNIFORM_LAYOUT._pad.offset, 0, true);
}

// ═══════════════════════════════════════════════════════════════════
// PIPELINE CACHE (one per model configuration)
// ═══════════════════════════════════════════════════════════════════

const pipelineCache = new Map<string, GPUComputePipeline>();

function getPipelineKey(c: KernelConstants): string {
  return `kernel_${c.HIDDEN_SIZE}_${c.HEAD_DIM}_${c.NUM_HEADS}`;
}

async function getPipeline(constants: KernelConstants): Promise<GPUComputePipeline> {
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

export async function runKernel(
  input: GPUBuffer,
  weights: GPUBuffer,
  output: GPUBuffer,
  constants: KernelConstants,
  uniforms: KernelUniforms,
): Promise<void> {
  const device = getDevice();
  const pipeline = await getPipeline(constants);

  const uniformBuffer = createUniformBuffer(UNIFORM_SIZE, (view) => {
    writeUniforms(view, uniforms);
  });

  // Workgroup calculation - document the formula!
  // Total threads = seqLen * NUM_HEADS
  // Workgroups = ceil(total / WORKGROUP_SIZE)
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

---

## Config Maps Over If/Else

### DON'T: Decision Trees

```typescript
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

### DO: Rule Arrays

```typescript
// GOOD - declarative, auditable, testable

interface VariantRule {
  match: MatchCondition;
  variant: string;
}

/** Matmul variant selection - first match wins */
const MATMUL_VARIANTS: VariantRule[] = [
  // Q4K fused paths
  { match: { bDtype: 'q4k', M: 1, hasSubgroups: true, N: { gt: 4096 } }, variant: 'q4_fused_multicol' },
  { match: { bDtype: 'q4k', M: 1, hasSubgroups: true }, variant: 'q4_fused' },
  { match: { bDtype: 'q4k', M: 1 }, variant: 'q4_fused_batched' },

  // GEMV paths (M=1)
  { match: { M: 1, hasSubgroups: true, N: { gt: 2048 } }, variant: 'gemv_subgroup_multicol' },
  { match: { M: 1, hasSubgroups: true }, variant: 'gemv_subgroup' },
  { match: { M: 1 }, variant: 'gemv' },

  // General matmul
  { match: { aDtype: 'f16', bDtype: 'f16' }, variant: 'f16' },
  { match: { bDtype: 'f16' }, variant: 'f16w_f32a' },
  { match: {}, variant: 'f32' },  // Default
];

function selectMatmulVariant(ctx: MatmulContext): string {
  const rule = MATMUL_VARIANTS.find(r => matchesRule(r.match, ctx));
  return rule?.variant ?? 'f32';
}
```

### Rule Matcher Utility

```typescript
// utils/rule-matcher.ts

type MatchValue = number | string | boolean | { gt?: number; lt?: number; eq?: number };
type MatchCondition = Record<string, MatchValue>;

export function matchesRule(rule: MatchCondition, context: Record<string, any>): boolean {
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

export function selectByRules<T>(
  rules: Array<{ match: MatchCondition; value: T }>,
  context: Record<string, any>,
  defaultValue: T,
): T {
  const rule = rules.find(r => matchesRule(r.match, context));
  return rule?.value ?? defaultValue;
}
```

---

## Config Tables

Centralize magic numbers into typed tables:

```typescript
// gpu/kernels/config-tables.ts

type GPUVendor = 'apple' | 'nvidia' | 'amd' | 'intel' | 'default';
type KernelType = 'matmul' | 'attention' | 'ffn' | 'norm' | 'rope';
type Precision = 'f32' | 'f16' | 'q4k';

/** Optimal workgroup sizes by vendor and kernel */
export const WORKGROUP_SIZES: Record<GPUVendor, Record<KernelType, number>> = {
  apple: { matmul: 128, attention: 32, ffn: 128, norm: 256, rope: 256 },
  nvidia: { matmul: 256, attention: 64, ffn: 256, norm: 256, rope: 256 },
  amd: { matmul: 64, attention: 64, ffn: 64, norm: 128, rope: 128 },
  intel: { matmul: 128, attention: 32, ffn: 128, norm: 128, rope: 128 },
  default: { matmul: 256, attention: 64, ffn: 256, norm: 256, rope: 256 },
};

/** Tile sizes by operation and precision */
export const TILE_SIZES: Record<KernelType, Partial<Record<Precision, number>>> = {
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
} as const;

/** Feature flag → kernel variant mapping */
export const FEATURE_VARIANTS: Record<string, Array<{ features: string[]; variant: string }>> = {
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

---

## Config Flow Layers

### Layer 1: Manifest → ModelConfig

```typescript
// config/model-config.ts

export interface ModelConfig {
  // Direct from manifest
  hiddenSize: number;
  numHeads: number;
  headDim: number;
  numKvHeads: number;
  intermediateSize: number;
  vocabSize: number;
  numLayers: number;

  // Derived
  kvHeadDim: number;
  isGQA: boolean;

  // Feature flags
  features: {
    scaleEmbeddings: boolean;
    rmsNormWeightOffset: boolean;
    sandwichNorm: boolean;
    dualRoPE: boolean;
  };

  // Device-tuned
  workgroupSize: number;
}

export function parseModelConfig(manifest: RDRRManifest, device: DeviceCapabilities): ModelConfig {
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

```typescript
// config/kernel-specs.ts

export interface KernelSpec {
  pipelineKey: string;
  constants: Record<string, number>;
  uniformSize: number;
  workgroups: (uniforms: Record<string, number>) => [number, number, number];
}

type KernelSpecFactory = (model: ModelConfig) => KernelSpec;

/** Kernel spec factories - pure functions from model config */
export const KERNEL_SPECS: Record<string, KernelSpecFactory> = {
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

```typescript
// gpu/kernel-executor.ts

export class KernelExecutor {
  private pipelines = new Map<string, GPUComputePipeline>();
  private specs: Record<string, KernelSpec>;
  private device: GPUDevice;

  constructor(model: ModelConfig, device: GPUDevice) {
    this.device = device;
    // Pre-compute all specs from model config (pure transformation)
    this.specs = Object.fromEntries(
      Object.entries(KERNEL_SPECS).map(([name, factory]) => [name, factory(model)])
    );
  }

  async dispatch(
    kernel: string,
    buffers: GPUBuffer[],
    uniforms: Record<string, number>,
  ): Promise<void> {
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

  private async getPipeline(spec: KernelSpec): Promise<GPUComputePipeline> {
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
  matmul.ts              # Main kernel wrapper
  matmul-utils.ts        # Shared utilities (if needed)
  fused-matmul-norm.ts   # Fused variant

config/
  model-config.ts        # ModelConfig type and parser
  kernel-specs.ts        # KernelSpec factories
  config-tables.ts       # WORKGROUP_SIZES, TILE_SIZES, etc.
```

### Types

```typescript
// PascalCase for interfaces/types
interface ModelConfig { }
interface KernelSpec { }
type GPUVendor = 'apple' | 'nvidia' | ...;

// camelCase for variables
const modelConfig: ModelConfig = ...;
const kernelSpec: KernelSpec = ...;
```

### Constants

```typescript
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

```typescript
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

```typescript
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

```typescript
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

```typescript
// BAD - uniform creation knows about manifest
function createUniforms(manifest: RDRRManifest) {
  view.setUint32(0, manifest.hidden_size);  // Wrong layer!
}

// GOOD - uniforms only know about runtime values
function createUniforms(uniforms: KernelUniforms) {
  view.setUint32(0, uniforms.seqLen);
}
```

---

## See Also

- [WGSL Style Guide](./WGSL_STYLE_GUIDE.md) - Shader conventions
- [Coding Guide](./CODING_GUIDE.md) - General patterns
