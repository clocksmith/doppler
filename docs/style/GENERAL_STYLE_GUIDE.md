# DOPPLER General Style Guide

General coding conventions and patterns for the DOPPLER codebase.

## Core Principles

1. **Config as Code** - Declarative maps over imperative logic
2. **Layered Configuration** - Each layer transforms the previous
3. **Pure Functions** - Config transformations should be pure
4. **Explicit over Implicit** - No magic, document everything

---

## Language Policy: JavaScript + Declaration Files

Doppler source code is **JavaScript** with **TypeScript declaration files** (.d.ts) for every module.

| File | Contains |
|------|----------|
| `module.js` | Clean code, no type annotations |
| `module.d.ts` | Comprehensive type definitions |

### Why JavaScript

| Reason | Explanation |
|--------|-------------|
| **Hot-swap architecture** | JS/WGSL/JSON artifacts swap at runtime without rebuild |
| **No generation quality difference** | No benchmarks show LLMs generate better TS than JS [1][2] |
| **Tests are the type system** | Comprehensive tests catch type errors pre-production |
| **Simpler toolchain** | No compile step between edit and run |

### Why .d.ts Files

| Reason | Explanation |
|--------|-------------|
| **Type context for agents** | Agents read .d.ts files directly—no JSDoc pollution in JS |
| **Consumer compatibility** | TypeScript users can import Doppler with full type safety |
| **Self-documenting** | Types describe interfaces without runtime cost |

### The Test Equivalence Principle

> With comprehensive tests, compile-time errors become test-time errors.
> Both block bad code before production. The difference is tooling, not safety.

### File Format Summary

| Format | Use For |
|--------|---------|
| **JavaScript (.js)** | All source code (clean, no type annotations) |
| **Declaration files (.d.ts)** | Type specs for every module |
| **JSON (.json)** | Static presets, manifests, fixtures |
| **WGSL (.wgsl)** | GPU shaders |

### Evidence

| Claim | Source |
|-------|--------|
| No JS vs TS generation quality difference | No benchmark measures this; both have equal training data [2] |
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

### Nullable Required Fields

For fields that can be legitimately disabled:

- `null` = explicitly disabled (valid)
- `undefined` = not specified (validation error)

Example: `slidingWindow: null` means "no sliding window" (intentional). Missing
`slidingWindow` means the manifest is incomplete (fail fast).

Example:
```json
{
  "attention": {
    "slidingWindow": null,
    "attnLogitSoftcapping": null
  }
}
```
Both fields are explicitly disabled (valid). Omitting either field is invalid.

### Manifest as Source of Truth

- Converter embeds all model-specific inference params in `manifest.json`
- Runtime never detects model family in pipeline code
- Pipeline reads config values directly, no architecture-string inference

```typescript
// DON'T: infer behavior from model family strings
if (arch.includes('gemma2')) useSoftcapping = true;

// DO: read config directly
const useSoftcapping = config.attnLogitSoftcapping !== null;
```

### Kernel Path Overrides

- Use `kernelPath` for kernel selection overrides.
- Manifest: `optimizations.kernelPath` or `inference.defaultKernelPath`
- Runtime: `runtime.inference.kernelPath`
- Legacy `kernelPlan` is removed; do not add new references.
- Precedence (low → high): manifest `optimizations.kernelPath` → manifest `inference.defaultKernelPath` → runtime config `runtime.inference.kernelPath` → CLI `--kernel-path`.

### Manifest-First Change Checklist

When adding a new inference knob or model behavior:
- Add it to `ManifestInferenceSchema` (and defaults for converter fixtures).
- Populate it in the converter (preset + HF config mapping).
- Merge it in `src/config/merge.ts` with source tracking.
- Validate it in `parseModelConfigFromManifest()` (null vs undefined rules).
- Add/extend tests that assert manifest values and override precedence.

### KV Cache Dtype Policy

- Default KV cache dtype to `f16` when supported.
- Force `f32` only when required (e.g., attention softcapping or no `shader-f16`).
- Do not introduce new defaults that silently upgrade to `f32`.

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

## Buffer Lifecycle

Buffer allocation uses **power-of-2 bucketing** with coarse-grained reuse rather than per-tensor allocation.

### Principles

1. **Bucket by size** - Round up to power-of-2 boundaries (256B, 512B, 1KB, 2KB, ...)
2. **Reuse aggressively** - Return buffers to pool instead of destroying
3. **Coarse granularity** - Fewer size classes = more reuse opportunities
4. **Explicit lifecycle** - Callers acquire and release buffers; no GC

### DON'T: Per-Tensor Allocation

```typescript
// BAD - creates fragmentation, no reuse
const buffer = device.createBuffer({ size: exactTensorSize });
// ... use buffer ...
buffer.destroy();
```

### DO: Pool Acquisition

```typescript
// GOOD - bucketed allocation with reuse
const buffer = bufferPool.acquire(exactTensorSize);
// ... use buffer ...
bufferPool.release(buffer);
```

### Why This Matters

GPU buffer creation is expensive (~1ms). Pooling with coarse buckets amortizes creation cost and reduces memory fragmentation. The slight memory waste from rounding up is far cheaper than allocation overhead.

---

## JSON-Driven Layer Plans

Pipeline step ordering can be overridden via JSON without code changes.

### Default Behavior

When no pipeline plan is specified, DOPPLER uses the optimized hardcoded layer path.

### Override Mechanism

Model presets may define `inference.pipeline` to drive per-layer step order. Runtime config can supply `runtime.inference.pipeline` for ad-hoc experiments.

```json
{
  "runtime": {
    "inference": {
      "pipeline": [
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
```

### When to Use

- Debugging layer ordering issues
- Testing new normalization patterns
- Comparing sandwich vs standard norm
- A/B testing fusion strategies

### When NOT to Use

For production inference, let the hardcoded path run—it's optimized for the common case.

---

## Loader Layer Configuration

The loader transforms raw weight files (GGUF/SafeTensors) into GPU buffers. It has its own config concerns separate from inference.

### Quantization Format Constants

Format-specific constants (block sizes, byte layouts) are **invariants** derived from the quantization spec. Import from a single source:

```typescript
// DON'T: Compute or redefine format constants locally
const blockBytes = 2 + 2 + K_SCALE_SIZE + QK_K / 2;  // Computing Q4K block size
const Q4K_BLOCK_BYTES = 144;  // Redefined locally

// DO: Import from quantization-constants.ts
import { Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES } from './quantization-constants.js';
```

### Dtype Defaults

Default dtypes for conversion should come from config, not hardcoded strings:

```typescript
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

Cache sizes and memory thresholds should come from config:

```typescript
// DON'T: Hardcoded memory limits
const maxCacheSize = 256 * 1024 * 1024;  // 256MB

// DO: Reference config schema
import { getStorageDefaults } from '../config/schema/index.js';
const maxCacheSize = getStorageDefaults().expertCache.maxSize;
```

---

## File Size Guidelines

| Threshold | Lines | Action |
|-----------|-------|--------|
| **Target** | 200-400 | Ideal file size |
| **Soft limit** | 750 | Consider splitting |
| **Hard limit** | 1000+ | Must split |

When splitting files:
- Extract cohesive functionality into separate modules
- Group by feature, not by type (e.g., `attention.ts` not `helpers.ts`)
- Keep related code together; don't split just to hit a number

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
├── formats/
│   └── rdrr/                  # Manifest parsing, RDRR types
│
└── storage/
    └── shard-manager.ts       # OPFS storage
```

---

## Naming Conventions

### Files

| Pattern | Example | Use For |
|---------|---------|---------|
| `kebab-case.js` | `model-config.js` | JavaScript modules |
| `snake_case.wgsl` | `matmul_f16.wgsl` | WGSL shaders |
| `UPPER_CASE.md` | `GENERAL_STYLE_GUIDE.md` | Documentation |

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
- [JavaScript Style Guide](./JAVASCRIPT_STYLE_GUIDE.md) - Kernel wrapper conventions
- [Kernel Compatibility](../KERNEL_COMPATIBILITY.md) - Runtime modes and flags
