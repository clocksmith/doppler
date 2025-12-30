# DOPPLER Kernel Style Guide

This document defines conventions for WebGPU kernels in DOPPLER.

## Overview

Kernels have three configuration layers:

| Layer | When Set | Example | Mechanism |
|-------|----------|---------|-----------|
| **Constants** | Pipeline creation | `hiddenSize`, `headDim` | Pipeline overrides |
| **Uniforms** | Per-dispatch | `seqLen`, `startPos` | Uniform buffer |
| **Bindings** | Per-dispatch | Input/output buffers | Bind group |

**Rule:** Values known at model load time are **constants**. Values that change per-inference are **uniforms**.

---

## 1. WGSL Structure

### File Layout

```wgsl
// kernel_name.wgsl

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS (override at pipeline creation)
// ═══════════════════════════════════════════════════════════════════

override HIDDEN_SIZE: u32 = 2048u;       // From manifest
override HEAD_DIM: u32 = 64u;            // From manifest
override NUM_HEADS: u32 = 32u;           // From manifest
override WORKGROUP_SIZE: u32 = 256u;     // Device-tuned

// ═══════════════════════════════════════════════════════════════════
// UNIFORMS (set per-dispatch)
// ═══════════════════════════════════════════════════════════════════

struct Uniforms {
    seq_len: u32,      // Changes every inference
    start_pos: u32,    // Changes every decode step
    kv_len: u32,       // Changes as KV cache grows
    _pad: u32,         // Alignment padding
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// ═══════════════════════════════════════════════════════════════════
// BINDINGS (buffers)
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// ═══════════════════════════════════════════════════════════════════
// ENTRY POINT
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Use HIDDEN_SIZE, HEAD_DIM as compile-time constants
    // Use u.seq_len, u.start_pos as runtime values
}
```

### Constant Categories

| Category | Examples | Source | When to Use |
|----------|----------|--------|-------------|
| **Model dims** | `HIDDEN_SIZE`, `HEAD_DIM`, `NUM_HEADS`, `INTERMEDIATE_SIZE` | manifest.json | Always constant per model |
| **Quant params** | `BLOCK_SIZE` (256 for Q4K) | Format spec | Always constant per format |
| **Device tune** | `WORKGROUP_SIZE`, `TILE_SIZE` | Device detection | Constant per device |
| **Feature flags** | `USE_CAUSAL_MASK`, `SCALE_EMBEDDINGS` | manifest.json | Constant per model |

### Uniform Categories

| Category | Examples | When Changes |
|----------|----------|--------------|
| **Position** | `seq_len`, `start_pos`, `kv_len` | Every decode step |
| **Batch** | `batch_size`, `num_tokens` | Every inference |
| **Runtime flags** | `is_prefill` | Prefill vs decode |

---

## 2. TypeScript Wrapper Structure

### File Layout

```typescript
// kernel-name.ts

import { getDevice } from '../device.js';
import { createPipelineWithConstants, createUniformBuffer } from './utils.js';
import { getModelConstants } from './constants.js';

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

/** Uniform struct - must match WGSL exactly */
interface KernelUniforms {
  seqLen: number;
  startPos: number;
  kvLen: number;
}

/** Constants baked into pipeline - from model manifest */
interface KernelConstants {
  HIDDEN_SIZE: number;
  HEAD_DIM: number;
  NUM_HEADS: number;
  WORKGROUP_SIZE: number;
}

// ═══════════════════════════════════════════════════════════════════
// UNIFORM BUFFER
// ═══════════════════════════════════════════════════════════════════

const UNIFORM_SIZE = 16;  // 4 u32 fields = 16 bytes

function writeUniforms(view: DataView, u: KernelUniforms): void {
  view.setUint32(0, u.seqLen, true);
  view.setUint32(4, u.startPos, true);
  view.setUint32(8, u.kvLen, true);
  view.setUint32(12, 0, true);  // padding
}

// ═══════════════════════════════════════════════════════════════════
// PIPELINE CREATION (with constants)
// ═══════════════════════════════════════════════════════════════════

const pipelineCache = new Map<string, GPUComputePipeline>();

export async function getKernelPipeline(
  constants: KernelConstants
): Promise<GPUComputePipeline> {
  const key = `${constants.HIDDEN_SIZE}_${constants.HEAD_DIM}_${constants.NUM_HEADS}`;

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
  const pipeline = await getKernelPipeline(constants);

  // Uniform buffer with runtime values
  const uniformBuffer = createUniformBuffer(UNIFORM_SIZE, (view) => {
    writeUniforms(view, uniforms);
  });

  // Dispatch
  const workgroups = Math.ceil(
    (uniforms.seqLen * constants.NUM_HEADS) / constants.WORKGROUP_SIZE
  );

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

## 3. Constants Flow: Manifest → Pipeline

### Model Loading

```typescript
// loader/doppler-loader.ts

export interface ModelConstants {
  // From manifest.json
  hiddenSize: number;
  headDim: number;
  numHeads: number;
  numKvHeads: number;
  intermediateSize: number;
  vocabSize: number;
  numLayers: number;

  // Derived
  kvHeadDim: number;  // hiddenSize / numKvHeads

  // Feature flags
  scaleEmbeddings: boolean;
  rmsNormWeightOffset: boolean;

  // Device-tuned (detected at init)
  workgroupSize: number;
  tileSize: number;
}

export function extractModelConstants(manifest: RDRRManifest): ModelConstants {
  return {
    hiddenSize: manifest.hidden_size,
    headDim: manifest.head_dim ?? manifest.hidden_size / manifest.num_attention_heads,
    numHeads: manifest.num_attention_heads,
    numKvHeads: manifest.num_key_value_heads ?? manifest.num_attention_heads,
    intermediateSize: manifest.intermediate_size,
    vocabSize: manifest.vocab_size,
    numLayers: manifest.num_hidden_layers,
    kvHeadDim: manifest.hidden_size / (manifest.num_key_value_heads ?? manifest.num_attention_heads),
    scaleEmbeddings: manifest.scale_embeddings ?? false,
    rmsNormWeightOffset: manifest.rms_norm_weight_offset ?? false,
    workgroupSize: getOptimalWorkgroupSize(),
    tileSize: getOptimalTileSize(),
  };
}
```

### Pipeline Registry

```typescript
// gpu/kernels/registry.ts

/**
 * Pipeline registry - one pipeline per (kernel, model) pair.
 * Constants are baked in at creation time.
 */
export class KernelRegistry {
  private pipelines = new Map<string, GPUComputePipeline>();
  private modelConstants: ModelConstants;

  constructor(constants: ModelConstants) {
    this.modelConstants = constants;
  }

  async getPipeline(kernel: KernelName): Promise<GPUComputePipeline> {
    const key = kernel;
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      pipeline = await this.createPipeline(kernel);
      this.pipelines.set(key, pipeline);
    }
    return pipeline;
  }

  private async createPipeline(kernel: KernelName): Promise<GPUComputePipeline> {
    const constants = this.getConstantsForKernel(kernel);
    return createPipelineWithConstants(kernel, 'main', constants);
  }

  private getConstantsForKernel(kernel: KernelName): Record<string, number> {
    const c = this.modelConstants;

    switch (kernel) {
      case 'attention':
        return {
          HEAD_DIM: c.headDim,
          NUM_HEADS: c.numHeads,
          NUM_KV_HEADS: c.numKvHeads,
          WORKGROUP_SIZE: c.workgroupSize,
        };
      case 'ffn':
        return {
          HIDDEN_SIZE: c.hiddenSize,
          INTERMEDIATE_SIZE: c.intermediateSize,
          WORKGROUP_SIZE: c.workgroupSize,
        };
      case 'rmsnorm':
        return {
          HIDDEN_SIZE: c.hiddenSize,
          WEIGHT_OFFSET: c.rmsNormWeightOffset ? 1 : 0,
          WORKGROUP_SIZE: c.workgroupSize,
        };
      // ... other kernels
    }
  }
}
```

---

## 4. Fused Kernels

### Decision Matrix

Fused kernels combine multiple operations. Use when:

| Fused Kernel | When to Use | Fallback |
|--------------|-------------|----------|
| `fused_matmul_rmsnorm` | Decode (M=1), N ≤ 8192 | matmul → rmsnorm |
| `fused_ffn` | Gate+Up+SiLU in one pass | matmul → silu → matmul |
| `fused_matmul_q4` | Q4K weights, aligned dims | dequant → matmul |

### Fused Kernel Structure

```wgsl
// fused_op.wgsl

// Same constant/uniform pattern
override HIDDEN_SIZE: u32 = 2048u;
override INTERMEDIATE_SIZE: u32 = 8192u;

struct Uniforms {
    seq_len: u32,
    // ... runtime values
}

// Multiple logical operations in one kernel
@compute @workgroup_size(256, 1, 1)
fn fused_ffn_swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 1. Gate projection
    let gate = matmul_row(input, gate_weights);

    // 2. Up projection
    let up = matmul_row(input, up_weights);

    // 3. SiLU activation
    let activated = silu(gate) * up;

    // 4. Down projection
    output[gid.x] = matmul_row(activated, down_weights);
}
```

### TypeScript Wrapper

```typescript
// fused-ffn.ts

export function shouldUseFusedFFN(
  seqLen: number,
  hiddenSize: number,
  intermediateSize: number
): boolean {
  // Only fuse when beneficial
  if (seqLen > 1) return false;  // Decode only
  if (intermediateSize > 16384) return false;  // Shared memory limit
  return true;
}

export async function runFFN(
  input: GPUBuffer,
  gateWeights: GPUBuffer,
  upWeights: GPUBuffer,
  downWeights: GPUBuffer,
  output: GPUBuffer,
  constants: ModelConstants,
  uniforms: { seqLen: number },
): Promise<void> {
  if (shouldUseFusedFFN(uniforms.seqLen, constants.hiddenSize, constants.intermediateSize)) {
    return runFusedFFN(input, gateWeights, upWeights, downWeights, output, constants, uniforms);
  }

  // Fallback: separate operations
  const intermediate = acquireBuffer(constants.intermediateSize * 4);
  await runMatmul(input, gateWeights, intermediate, ...);
  await runSiLU(intermediate, ...);
  await runMatmul(intermediate, downWeights, output, ...);
  releaseBuffer(intermediate);
}
```

---

## 5. Naming Conventions

### WGSL Files

```
{operation}_{variant}.wgsl

Examples:
  matmul_f16.wgsl        # F16 matmul
  matmul_gemv.wgsl       # M=1 optimized
  attention_causal.wgsl  # With causal mask
  fused_ffn_swiglu.wgsl  # Fused FFN
```

### TypeScript Files

```
{operation}.ts           # Main wrapper
{operation}-utils.ts     # Shared utilities (if needed)
fused-{operation}.ts     # Fused variant

Examples:
  matmul.ts
  attention.ts
  fused-ffn.ts
```

### Constants

```
UPPER_SNAKE_CASE for WGSL overrides
camelCase for TypeScript

WGSL:    override HIDDEN_SIZE: u32 = 2048u;
TS:      constants.hiddenSize
```

### Entry Points

```
main                     # Default entry
{operation}_{variant}    # Variant-specific

Examples:
  @compute fn main(...)
  @compute fn gemv_rmsnorm_small(...)
  @compute fn attention_causal(...)
```

---

## 6. Checklist for New Kernels

- [ ] **Constants identified** - Which values are model-specific?
- [ ] **Uniforms identified** - Which values change per-dispatch?
- [ ] **Uniform struct matches** - WGSL struct == TypeScript writer
- [ ] **Pipeline caching** - One pipeline per model, not per-dispatch
- [ ] **Workgroup size tunable** - Use `override WORKGROUP_SIZE`
- [ ] **Dispatch calculation documented** - Comment the formula
- [ ] **Fused variant considered** - Is fusion beneficial?
- [ ] **Fallback path exists** - For unsupported cases

---

## 7. Config-as-Code (JavaScript Guide)

### Mental Model

```
Model Manifest → JS Config → Pipeline Config → Kernel Config → Kernel Execution
     ↓              ↓              ↓               ↓              ↓
  manifest.json  ModelConfig   PipelineSpec   KernelSpec    GPUComputePass
```

Each layer transforms the previous into a more specific configuration.

### DON'T: Decision Trees

```typescript
// BAD - 70 lines of nested if/else
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
}
```

### DO: Config Maps

```typescript
// GOOD - declarative config, easy to audit

/** Matmul variant selection rules */
const MATMUL_VARIANTS: VariantRule[] = [
  // Q4K fused paths (order matters - first match wins)
  {
    match: { bDtype: 'q4k', M: 1, hasSubgroups: true, N: { gt: 4096 } },
    variant: 'q4_fused_multicol',
  },
  {
    match: { bDtype: 'q4k', M: 1, hasSubgroups: true },
    variant: 'q4_fused',
  },
  {
    match: { bDtype: 'q4k', M: 1 },
    variant: 'q4_fused_batched',
  },

  // GEMV paths (M=1)
  {
    match: { M: 1, hasSubgroups: true, N: { gt: 2048 } },
    variant: 'gemv_subgroup_multicol',
  },
  {
    match: { M: 1, hasSubgroups: true },
    variant: 'gemv_subgroup',
  },
  {
    match: { M: 1 },
    variant: 'gemv',
  },

  // General matmul
  {
    match: { aDtype: 'f16', bDtype: 'f16' },
    variant: 'f16',
  },
  {
    match: { bDtype: 'f16' },
    variant: 'f16w_f32a',
  },
  {
    match: {},  // Default
    variant: 'f32',
  },
];

function selectMatmulVariant(ctx: MatmulContext): string {
  const rule = MATMUL_VARIANTS.find(r => matchesRule(r.match, ctx));
  return rule?.variant ?? 'f32';
}
```

### Config Layers

#### Layer 1: Model Manifest → ModelConfig

```typescript
// manifest.json (raw)
{
  "hidden_size": 2048,
  "num_attention_heads": 32,
  "head_dim": 64,
  "rms_norm_weight_offset": true,
  ...
}

// ModelConfig (typed, validated, derived values added)
interface ModelConfig {
  // Direct from manifest
  hiddenSize: number;
  numHeads: number;
  headDim: number;

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
}

function parseModelConfig(manifest: RDRRManifest): ModelConfig {
  return {
    hiddenSize: manifest.hidden_size,
    numHeads: manifest.num_attention_heads,
    headDim: manifest.head_dim ?? manifest.hidden_size / manifest.num_attention_heads,
    kvHeadDim: manifest.hidden_size / (manifest.num_key_value_heads ?? manifest.num_attention_heads),
    isGQA: (manifest.num_key_value_heads ?? manifest.num_attention_heads) !== manifest.num_attention_heads,
    features: {
      scaleEmbeddings: manifest.scale_embeddings ?? false,
      rmsNormWeightOffset: manifest.rms_norm_weight_offset ?? false,
      sandwichNorm: !!manifest.pre_feedforward_layernorm,
      dualRoPE: !!manifest.rope_local_base_freq,
    },
  };
}
```

#### Layer 2: ModelConfig → PipelineSpec

```typescript
// Pipeline specification - what operations to run
interface PipelineSpec {
  embed: EmbedSpec;
  layers: LayerSpec[];
  logits: LogitsSpec;
}

interface LayerSpec {
  attention: AttentionSpec;
  ffn: FFNSpec;
  norms: NormSpec[];
}

interface AttentionSpec {
  variant: 'standard' | 'gqa' | 'mqa';
  kernels: {
    qkv: KernelRef;
    rope: KernelRef;
    attn: KernelRef;
    proj: KernelRef;
  };
  ropeConfig: {
    theta: number;
    localTheta?: number;  // For dual RoPE
  };
}

// Build pipeline spec from model config
function buildPipelineSpec(model: ModelConfig, device: DeviceCapabilities): PipelineSpec {
  return {
    embed: {
      kernel: 'embed',
      scale: model.features.scaleEmbeddings ? Math.sqrt(model.hiddenSize) : 1.0,
    },
    layers: Array.from({ length: model.numLayers }, (_, i) => ({
      attention: buildAttentionSpec(model, device, i),
      ffn: buildFFNSpec(model, device),
      norms: buildNormSpecs(model, i),
    })),
    logits: {
      kernel: selectLogitsKernel(model, device),
    },
  };
}
```

#### Layer 3: PipelineSpec → KernelSpec

```typescript
// Kernel specification - exact parameters for GPU dispatch
interface KernelSpec {
  pipeline: string;          // Cached pipeline key
  constants: Record<string, number>;
  uniformSize: number;
  workgroups: (uniforms: any) => [number, number, number];
}

// Kernel specs derived from model config
const KERNEL_SPECS: Record<string, (model: ModelConfig) => KernelSpec> = {
  attention_standard: (model) => ({
    pipeline: `attention_standard_h${model.numHeads}_d${model.headDim}`,
    constants: {
      NUM_HEADS: model.numHeads,
      HEAD_DIM: model.headDim,
      NUM_KV_HEADS: model.numKvHeads,
    },
    uniformSize: 16,
    workgroups: (u) => [model.numHeads, Math.ceil(u.seqLen / 32), 1],
  }),

  ffn_swiglu: (model) => ({
    pipeline: `ffn_swiglu_h${model.hiddenSize}_i${model.intermediateSize}`,
    constants: {
      HIDDEN_SIZE: model.hiddenSize,
      INTERMEDIATE_SIZE: model.intermediateSize,
    },
    uniformSize: 8,
    workgroups: (u) => [Math.ceil(u.seqLen * model.intermediateSize / 256), 1, 1],
  }),

  rmsnorm: (model) => ({
    pipeline: `rmsnorm_h${model.hiddenSize}_o${model.features.rmsNormWeightOffset ? 1 : 0}`,
    constants: {
      HIDDEN_SIZE: model.hiddenSize,
      WEIGHT_OFFSET: model.features.rmsNormWeightOffset ? 1 : 0,
    },
    uniformSize: 4,
    workgroups: (u) => [u.seqLen, 1, 1],
  }),
};
```

#### Layer 4: KernelSpec → Execution

```typescript
// Kernel executor - uses specs to dispatch
class KernelExecutor {
  private pipelines = new Map<string, GPUComputePipeline>();
  private specs: Record<string, KernelSpec>;

  constructor(model: ModelConfig) {
    // Pre-compute all kernel specs from model config
    this.specs = Object.fromEntries(
      Object.entries(KERNEL_SPECS).map(([name, specFn]) => [name, specFn(model)])
    );
  }

  async dispatch(
    kernelName: string,
    buffers: GPUBuffer[],
    uniforms: Record<string, number>,
  ): Promise<void> {
    const spec = this.specs[kernelName];
    const pipeline = await this.getPipeline(spec);
    const uniformBuffer = this.createUniformBuffer(spec.uniformSize, uniforms);
    const workgroups = spec.workgroups(uniforms);

    // Single dispatch path for all kernels
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.createBindGroup(pipeline, uniformBuffer, buffers));
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
```

### Config Tables

Replace scattered constants with centralized tables:

```typescript
// gpu/kernels/config-tables.ts

/** Workgroup sizes by device vendor */
export const WORKGROUP_SIZES: Record<GPUVendor, Record<KernelType, number>> = {
  apple: {
    matmul: 128,
    attention: 32,
    ffn: 128,
    norm: 256,
  },
  nvidia: {
    matmul: 256,
    attention: 64,
    ffn: 256,
    norm: 256,
  },
  amd: {
    matmul: 64,
    attention: 64,
    ffn: 64,
    norm: 128,
  },
  default: {
    matmul: 256,
    attention: 64,
    ffn: 256,
    norm: 256,
  },
};

/** Tile sizes by operation and precision */
export const TILE_SIZES: Record<string, Record<Precision, number>> = {
  matmul: {
    f32: 16,
    f16: 16,
    q4k: 32,
  },
  attention: {
    f32: 32,
    f16: 64,
  },
};

/** Fusion thresholds - when to use fused kernels */
export const FUSION_THRESHOLDS = {
  matmul_rmsnorm: {
    maxM: 1,        // Decode only
    maxN: 8192,     // Shared memory limit
  },
  ffn_swiglu: {
    maxSeqLen: 1,   // Decode only
    maxIntermediate: 16384,
  },
  attention_flash: {
    minSeqLen: 128, // Only worth it for long sequences
  },
};

/** Feature detection → kernel variant mapping */
export const FEATURE_VARIANTS = {
  attention: {
    conditions: [
      { features: ['gqa', 'causal'], variant: 'attention_gqa_causal' },
      { features: ['gqa'], variant: 'attention_gqa' },
      { features: ['causal'], variant: 'attention_causal' },
      { features: [], variant: 'attention_standard' },
    ],
  },
  rope: {
    conditions: [
      { features: ['dualRoPE'], variant: 'rope_dual' },
      { features: [], variant: 'rope_standard' },
    ],
  },
};
```

### Utility: Rule Matcher

```typescript
// utils/rule-matcher.ts

interface MatchRule {
  [key: string]: number | string | boolean | { gt?: number; lt?: number; eq?: number };
}

function matchesRule(rule: MatchRule, context: Record<string, any>): boolean {
  for (const [key, expected] of Object.entries(rule)) {
    const actual = context[key];

    if (typeof expected === 'object' && expected !== null) {
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

function selectByRules<T>(
  rules: Array<{ match: MatchRule; value: T }>,
  context: Record<string, any>,
  defaultValue: T,
): T {
  const rule = rules.find(r => matchesRule(r.match, context));
  return rule?.value ?? defaultValue;
}
```

---

## 8. Anti-Patterns

### DON'T: Pass model dimensions as uniforms

```wgsl
// BAD - hiddenSize is constant, wastes uniform bandwidth
struct Uniforms {
    hidden_size: u32,  // Should be override constant!
    seq_len: u32,
}
```

### DON'T: Hardcode workgroup sizes

```wgsl
// BAD - can't tune for different devices
@compute @workgroup_size(256, 1, 1)
fn main(...) { }

// GOOD - tunable
override WORKGROUP_SIZE: u32 = 256u;
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(...) { }
```

### DON'T: Create pipeline per-dispatch

```typescript
// BAD - pipeline creation is expensive
async function runKernel() {
  const pipeline = await createPipeline(...);  // Every time!
}

// GOOD - cache and reuse
const pipelineCache = new Map();
async function runKernel() {
  const pipeline = pipelineCache.get(key) ?? await createPipeline(...);
}
```

### DON'T: Hardcode uniform buffer sizes

```typescript
// BAD - magic number, easy to mismatch
const uniformBuffer = createBuffer(32);

// GOOD - derived from struct
const UNIFORMS = {
  seqLen: 4,
  startPos: 4,
  kvLen: 4,
  _pad: 4,
} as const;
const UNIFORM_SIZE = Object.values(UNIFORMS).reduce((a, b) => a + b, 0);
```
