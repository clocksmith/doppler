# DOPPLER Coding Guide

General coding conventions and patterns for the DOPPLER codebase.

## Core Principles

1. **Config as Code** - Declarative maps over imperative logic
2. **Layered Configuration** - Each layer transforms the previous
3. **Pure Functions** - Config transformations should be pure
4. **Explicit over Implicit** - No magic, document everything

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

### Layer 2: ModelConfig (TypeScript)

Typed, validated, with derived values:

```typescript
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

### Layer 3: PipelineSpec (What to run)

Sequence of operations:

```typescript
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

### Layer 4: KernelSpec (How to run)

GPU dispatch parameters:

```typescript
interface KernelSpec {
  pipelineKey: string;                    // Cache key
  constants: Record<string, number>;      // Baked into pipeline
  uniformSize: number;                    // Bytes
  workgroups: WorkgroupFn;                // Dispatch calculation
}

type WorkgroupFn = (uniforms: Record<string, number>) => [number, number, number];
```

### Layer 5: Runtime Uniforms (Per-dispatch)

Values that change each inference:

```typescript
interface RuntimeUniforms {
  seqLen: number;      // Tokens this pass
  startPos: number;    // Position for RoPE
  kvLen: number;       // KV cache length
  isPrefill: boolean;  // Prefill vs decode
}
```

---

## File Organization

```
doppler/
├── config/                    # Configuration layer
│   ├── model-config.ts        # ModelConfig type and parser
│   ├── pipeline-spec.ts       # PipelineSpec builder
│   ├── kernel-specs.ts        # KernelSpec factories
│   └── config-tables.ts       # WORKGROUP_SIZES, THRESHOLDS, etc.
│
├── gpu/
│   ├── device.ts              # WebGPU device management
│   ├── buffer-pool.ts         # Buffer allocation
│   └── kernels/
│       ├── utils.ts           # Pipeline creation, bind groups
│       ├── kernel-executor.ts # Unified dispatch
│       ├── matmul.ts          # Matmul variants
│       ├── matmul_f16.wgsl
│       ├── matmul_gemv.wgsl
│       ├── attention.ts
│       ├── attention_causal.wgsl
│       └── ...
│
├── inference/
│   ├── pipeline.ts            # Main inference loop
│   └── pipeline/
│       ├── embed.ts           # Embedding layer
│       ├── layer.ts           # Transformer layer
│       ├── attention.ts       # Attention computation
│       ├── ffn.ts             # Feed-forward network
│       └── logits.ts          # Output projection
│
├── loader/
│   ├── doppler-loader.ts      # Model loading
│   └── weights.ts             # Weight types
│
└── storage/
    ├── shard-manager.ts       # OPFS storage
    └── rdrr-format.ts         # Manifest parsing
```

---

## Naming Conventions

### Files

| Pattern | Example | Use For |
|---------|---------|---------|
| `kebab-case.ts` | `model-config.ts` | TypeScript modules |
| `snake_case.wgsl` | `matmul_f16.wgsl` | WGSL shaders |
| `UPPER_CASE.md` | `CODING_GUIDE.md` | Documentation |

### Types

```typescript
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

```typescript
// camelCase for variables and functions
const modelConfig = parseModelConfig(manifest);
const kernelSpec = KERNEL_SPECS.attention(modelConfig);

function selectMatmulVariant(ctx: MatmulContext): string { }
```

### Constants

```typescript
// UPPER_SNAKE_CASE for module-level constants
const WORKGROUP_SIZES = { ... };
const TILE_SIZES = { ... };
const FUSION_THRESHOLDS = { ... };

// camelCase for derived/local constants
const workgroupSize = WORKGROUP_SIZES[vendor].matmul;
```

### WGSL

```wgsl
// UPPER_SNAKE_CASE for constants
override HIDDEN_SIZE: u32 = 2048u;
override WORKGROUP_SIZE: u32 = 256u;

// snake_case for struct fields and functions
struct Uniforms {
    seq_len: u32,
    start_pos: u32,
}

fn compute_attention_score(q: vec4<f32>, k: vec4<f32>) -> f32 { }
```

---

## Error Handling

### Validation Errors (Fail Fast)

```typescript
function parseModelConfig(manifest: RDRRManifest): ModelConfig {
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

```typescript
async function dispatch(kernel: string, ...): Promise<void> {
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

```typescript
async function runKernel(...): Promise<void> {
  const device = getDevice();
  if (device.lost) {
    throw new Error('GPU device lost. Call initDevice() to recover.');
  }
  // ...
}
```

---

## Logging

Use the unified debug system:

```typescript
import { log, trace } from '../debug/index.js';

// Log levels (verbosity)
log.error('Module', 'Critical failure', { details });
log.warn('Module', 'Something unexpected');
log.info('Module', 'Normal operation');
log.verbose('Module', 'Detailed info');
log.debug('Module', 'Debug info');

// Trace categories (what to show)
trace.loader('Loading shard 0');
trace.kernels('matmul M=1 K=1152 N=1024');
trace.attn(layerIdx, 'Q maxAbs=1.2');
```

---

## Testing

### Unit Tests (Kernel Correctness)

```typescript
// kernel-tests/tests/matmul.test.ts

describe('matmul', () => {
  it('f16 produces correct output', async () => {
    const A = createTestBuffer([4, 8], 'f16');
    const B = createTestBuffer([8, 4], 'f16');
    const C = await runMatmul(A, B, 4, 4, 8, { variant: 'f16' });

    expect(await readBuffer(C)).toBeCloseTo(expectedOutput, 1e-3);
  });

  it('handles non-aligned dimensions', async () => {
    // K=1152 is not divisible by typical tile sizes
    const C = await runMatmul(A, B, 1, 2048, 1152);
    expect(await readBuffer(C)).toBeCloseTo(expectedOutput, 1e-3);
  });
});
```

### Config Tests (Rule Coverage)

```typescript
// config/matmul-variants.test.ts

describe('MATMUL_VARIANTS', () => {
  it('selects q4_fused for Q4K decode with subgroups', () => {
    const ctx = { bDtype: 'q4k', M: 1, hasSubgroups: true, N: 2048 };
    expect(selectMatmulVariant(ctx)).toBe('q4_fused');
  });

  it('falls back to f32 when no rules match', () => {
    const ctx = { bDtype: 'unknown', M: 4, hasSubgroups: false, N: 512 };
    expect(selectMatmulVariant(ctx)).toBe('f32');
  });
});
```

### Integration Tests (E2E)

```typescript
// tests/inference.test.ts

describe('inference', () => {
  it('generates coherent output', async () => {
    const pipeline = await createPipeline(modelPath);
    const output = await pipeline.generate('The color of the sky is', { maxTokens: 10 });

    expect(output).toContain('blue');
  });
});
```

---

## Documentation

### Code Comments

```typescript
// WHY, not WHAT
// BAD: Increment i
// GOOD: Skip padding tokens (they have token_id = 0)

// Document non-obvious behavior
/**
 * Workgroup calculation for attention kernel.
 *
 * We dispatch one workgroup per attention head. Each workgroup
 * processes all query positions for that head. This maximizes
 * cache locality for the KV values.
 *
 * Total workgroups: numHeads
 * Threads per workgroup: min(seqLen, WORKGROUP_SIZE)
 */
function getAttentionWorkgroups(numHeads: number, seqLen: number): [number, number, number] {
  return [numHeads, 1, 1];
}
```

### Config Documentation

```typescript
/** Matmul variant selection rules.
 *
 * Rules are evaluated in order - first match wins.
 *
 * Variants:
 * - q4_fused_*: Direct Q4K dequant + matmul (fastest for Q4K)
 * - gemv_*: M=1 optimized (decode path)
 * - f16*: F16 compute (when supported)
 * - f32: Fallback
 */
const MATMUL_VARIANTS: VariantRule[] = [
  // ...
];
```

---

## Anti-Patterns

### DON'T: Mix Configuration Layers

```typescript
// BAD - kernel dispatch knows about manifest
async function runAttention(manifest: RDRRManifest, ...) {
  const numHeads = manifest.num_attention_heads;  // Wrong layer!
}

// GOOD - kernel dispatch uses KernelSpec
async function runAttention(spec: KernelSpec, uniforms: RuntimeUniforms, ...) {
  const workgroups = spec.workgroups(uniforms);
}
```

### DON'T: Duplicate Configuration

```typescript
// BAD - same threshold in multiple places
if (N > 4096) { /* in matmul.ts */ }
if (N > 4096) { /* in gemv.ts */ }

// GOOD - single source of truth
const THRESHOLDS = { multicol: { minN: 4096 } };
if (N > THRESHOLDS.multicol.minN) { ... }
```

### DON'T: Implicit Defaults

```typescript
// BAD - hidden default
function runKernel(opts: { workgroupSize?: number }) {
  const wgSize = opts.workgroupSize || 256;  // Where does 256 come from?
}

// GOOD - explicit from config
function runKernel(spec: KernelSpec) {
  const wgSize = spec.constants.WORKGROUP_SIZE;  // From model config
}
```

---

## See Also

- [WGSL Style Guide](./WGSL_STYLE_GUIDE.md) - Shader conventions
- [TypeScript Style Guide](./TYPESCRIPT_STYLE_GUIDE.md) - Kernel wrapper conventions
- [Kernel Compatibility](../KERNEL_COMPATIBILITY.md) - Runtime modes and flags
