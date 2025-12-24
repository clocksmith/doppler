# Kernel Configuration System

**Status:** Phase 1 Complete (Manifest Integration)
**Prerequisites:** Phase 1 kernel infrastructure complete
**Goal:** Declarative configuration for kernel dispatch, with manifest defaults and runtime overrides.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Kernel Configuration Flow                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │  Conversion  │ ──> │   Manifest   │ ──> │   Pipeline Load      │ │
│  │  convert-cli │     │  (defaults)  │     │   setKernelHints()   │ │
│  └──────────────┘     └──────────────┘     └──────────┬───────────┘ │
│                                                       │              │
│                       ┌───────────────────────────────┼───────────┐ │
│                       │        Override Chain         │           │ │
│                       │                               ▼           │ │
│  ┌──────────────┐     │     ┌──────────────┐    ┌──────────────┐ │ │
│  │ YAML Profile │ ────┼──>  │ kernel-hints │ <──│  Runtime API │ │ │
│  │   (future)   │     │     │   module     │    │  (highest)   │ │ │
│  └──────────────┘     │     └──────┬───────┘    └──────────────┘ │ │
│                       └────────────┼─────────────────────────────┘ │
│                                    │                                │
│                                    ▼                                │
│                       ┌──────────────────────┐                     │
│                       │   matmul.ts / etc    │                     │
│                       │   isFusedQ4KDisabled │                     │
│                       │   shouldUseFusedQ4K  │                     │
│                       └──────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Override Priority** (highest to lowest):
1. **Runtime API** - `setKernelHints(hints, 'runtime')`
2. **YAML Profile** - `setKernelHints(hints, 'profile')` (future)
3. **Manifest** - `optimizations.kernelHints` (set during conversion)
4. **Built-in heuristics** - When no hints provided

---

## Implemented (Phase 1)

### TypeScript Types (`storage/rdrr-format.ts`)

```typescript
export type MatmulKernel = 'auto' | 'fused_q4k' | 'dequant_f16' | 'dequant_f32' | 'gemv_subgroup';
export type AttentionKernel = 'auto' | 'tiled_large' | 'tiled_small' | 'streaming';
export type Q4KLayout = 'flat' | 'row_wise' | 'column_wise';
export type ComputePrecision = 'f16' | 'f32' | 'auto';  // WebLLM-style compute precision

export interface KernelHints {
  computePrecision?: ComputePrecision;  // Global compute precision (like q4f16 vs q4f32)
  q4kMatmul?: MatmulKernel;             // Q4K weight matmul strategy
  f16Matmul?: MatmulKernel;             // F16 weight matmul strategy
  attentionPrefill?: AttentionKernel;
  attentionDecode?: AttentionKernel;
  tunedDevice?: string;                 // Device hints were tuned for
  benchmarkTokPerSec?: number;          // Performance achieved
}

export interface ConversionInfo {
  source: string;                  // HuggingFace ID or path
  convertedAt: string;             // ISO 8601 timestamp
  converterVersion: string;
  command?: string;                // CLI command for reproducibility
  quantization: {
    type: string;                  // "Q4_K_M", "F16", etc.
    layout?: Q4KLayout;
    fuseGateUp?: boolean;
  };
  originalDtype?: string;
}

export interface RuntimeOptimizations {
  attentionKernel?: AttentionKernel;
  kernelHints?: KernelHints;
}
```

### Manifest Structure

```json
{
  "version": "1.0",
  "modelId": "gemma-3-1b-it-q4-f16",
  "quantization": "Q4_K_M",

  "conversion": {
    "source": "google/gemma-3-1b-it",
    "convertedAt": "2025-12-18T15:30:00Z",
    "converterVersion": "1.0.0",
    "command": "npx tsx convert-cli.ts ... --q4k-layout column_wise --compute-precision f16",
    "quantization": {
      "type": "Q4_K_M",
      "layout": "column_wise",
      "fuseGateUp": false
    }
  },

  "optimizations": {
    "kernelHints": {
      "computePrecision": "f16",
      "q4kMatmul": "dequant_f16",
      "f16Matmul": "gemv_subgroup",
      "attentionPrefill": "tiled_large",
      "attentionDecode": "streaming"
    }
  }
}
```

### Kernel Hints Module (`gpu/kernel-hints.ts`)

```typescript
// Set hints from manifest or runtime
setKernelHints(hints: KernelHints, source: 'manifest' | 'profile' | 'runtime'): void

// Get current hints
getKernelHints(): KernelHints | null

// Check Q4K strategy
shouldUseFusedQ4K(): boolean  // Returns false if hints say 'dequant_*'

// Clear hints (on model unload)
clearKernelHints(): void

// Compute precision helpers (WebLLM-style)
getComputePrecision(): 'f16' | 'f32' | 'auto'
shouldUseF16Compute(hasShaderF16: boolean): boolean
getQ4KDequantStrategy(hasShaderF16: boolean): 'dequant_f16' | 'dequant_f32'
```

### Pipeline Integration (`inference/pipeline.ts`)

```typescript
async loadModel(manifest: any): Promise<void> {
  // ...
  const kernelHints = manifest.optimizations?.kernelHints;
  if (kernelHints) {
    setKernelHints(kernelHints, 'manifest');
  }
  // ...
}
```

### Matmul Integration (`gpu/kernels/matmul.ts`)

```typescript
export function isFusedQ4KDisabled(): boolean {
  // 1. Check window.DOPPLER_DISABLE_FUSED_Q4K (debug)
  // 2. Check kernel hints from manifest
  // 3. Default: false (use dequant path - 2x faster)
  return !shouldUseFusedQ4K();
}
```

---

## Current Default Hints (Based on Benchmarks - Dec 2025)

| Setting | Default | Reason |
|---------|---------|--------|
| `q4kLayout` | **`column_wise`** | **14% faster than flat, 2.7x faster than row_wise** |
| `computePrecision` | `auto` | Detect GPU capabilities at runtime |
| Q4K matmul | `dequant_f16` | Fused kernel has poor thread utilization |
| F16 | `gemv_subgroup` | Best for GEMV decode with subgroup operations |
| Attention prefill | `tiled_large` | Better for long sequences |
| Attention decode | `streaming` | Better for single-token generation |

### Benchmark Results (Gemma 3 1B, M3 MacBook)

| Variant | tok/s | vs Baseline |
|---------|-------|-------------|
| **Q4K column_wise** | **8.0** | +14% |
| Q4K flat | 7.0 | baseline |
| Q4K row_wise (fused) | 3.0 | -57% |
| F16 | 9.4 | +34% (2x VRAM) |

### WebLLM Naming Equivalents

| DOPPLER | WebLLM | CLI |
|---------|--------|-----|
| `gemma-3-1b-it-q4` | (auto detect) | `--compute-precision auto` |
| `gemma-3-1b-it-q4-f16` | `q4f16_1` | `--compute-precision f16` |
| `gemma-3-1b-it-q4-f32` | `q4f32_1` | `--compute-precision f32` |

---

## Milestones

### Phase 1: Manifest Integration (COMPLETE)

| Task | Status | Files |
|------|--------|-------|
| KernelHints TypeScript types | ✅ Done | `storage/rdrr-format.ts` |
| ConversionInfo types | ✅ Done | `storage/rdrr-format.ts` |
| Converter writes metadata | ✅ Done | `tools/convert-cli.ts` |
| Writer accepts conversion/optimizations | ✅ Done | `tools/rdrr-writer.ts` |
| Kernel hints module | ✅ Done | `gpu/kernel-hints.ts` |
| Pipeline reads manifest hints | ✅ Done | `inference/pipeline.ts` |
| Matmul uses hints | ✅ Done | `gpu/kernels/matmul.ts` |

### Phase 2: YAML Profiles (Future)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| YAML parser integration | P1 | ⬜ TODO | Use `js-yaml` |
| Profile loader | P1 | ⬜ TODO | `config/profile-loader.ts` |
| Device fingerprinting | P2 | ⬜ TODO | Auto-detect GPU |
| Profile caching | P2 | ⬜ TODO | localStorage/OPFS |

### Phase 3: Auto-Tuning (Future)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Benchmark harness | P1 | ⬜ TODO | Time kernel combinations |
| Profile generator | P1 | ⬜ TODO | Output best config |
| Per-device profiles | P2 | ⬜ TODO | Ship with package |

### Phase 4: Parallel Kernel Execution (Future)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| `recordParallelGroup()` method | P2 | ⬜ TODO | `gpu/command-recorder.ts` |
| FFN parallel gate/up | P2 | ⬜ TODO | Independent matmuls |
| Attention parallel QKV | P2 | ⬜ TODO | Independent projections |

---

## YAML Profile Schema (Future)

```yaml
# config/profiles/gemma3-1b-apple-m3.yaml
name: gemma3-1b-m3-optimized
model: gemma3-1b
version: 1

device:
  vendor: apple
  minSharedMemory: 32768
  requiresF16: true
  requiresSubgroups: true

kernelHints:
  q4kMatmul: dequant_f16      # Override manifest default
  f16Matmul: gemv_subgroup
  attentionPrefill: tiled_large
  attentionDecode: streaming
  tunedDevice: "Apple M3 Pro"
  benchmarkTokPerSec: 11.5

# Per-layer overrides (optional)
layers:
  0-5:
    attention: tiled_small
  6-25:
    attention: tiled_large
```

---

## API Usage

### Current (Phase 1)

```typescript
// Automatic: Pipeline reads hints from manifest on loadModel()
const pipeline = await createPipeline(manifest);
// Hints are automatically applied via gpu/kernel-hints.ts

// Manual override (highest priority)
import { setKernelHints } from './gpu/kernel-hints.js';
setKernelHints({
  q4kMatmul: 'fused_q4k',  // Force fused path
}, 'runtime');
```

### Future (Phase 2+)

```typescript
// Load device-specific profile
const profile = await loadProfile('gemma3-1b-apple-m3');
setKernelHints(profile.kernelHints, 'profile');

// Auto-tune and save
const bestProfile = await autoTune(pipeline, { iterations: 10 });
await saveProfile(bestProfile, 'my-device-optimized');
```

---

## Files Summary

| File | Status | Purpose |
|------|--------|---------|
| `storage/rdrr-format.ts` | ✅ Modified | KernelHints, ConversionInfo types |
| `tools/convert-cli.ts` | ✅ Modified | Write conversion metadata |
| `tools/rdrr-writer.ts` | ✅ Modified | Accept conversion/optimizations |
| `gpu/kernel-hints.ts` | ✅ Created | Kernel hints state management |
| `inference/pipeline.ts` | ✅ Modified | Load hints from manifest |
| `gpu/kernels/matmul.ts` | ✅ Modified | Use hints for Q4K selection |
| `config/profile-loader.ts` | ⬜ Future | YAML profile loading |
| `config/profiles/*.yaml` | ⬜ Future | Pre-tuned device profiles |
| `tools/auto-tune.ts` | ⬜ Future | Benchmark + generate profiles |

---

## Rationale

**Why manifest defaults + YAML overrides?**
- Manifest captures conversion-time decisions (quantization, layout)
- YAML profiles capture device-specific tuning
- Both are declarative and reproducible
- No need to rebuild model for different devices

**Why default to column_wise layout?**
- Dec 2025 benchmarks show **8 tok/s** (14% faster than flat's 7 tok/s)
- **2.7x faster** than row_wise with fused kernel (row_wise: 3 tok/s)
- Column-wise packing gives coalesced memory access for GEMV decode
- Each output column's Q4K blocks are contiguous → high bandwidth

**Why default to dequant_f16 for Q4K?**
- Benchmarks show 2.3x faster than fused kernel (8 tok/s vs 3 tok/s)
- Fused kernel has poor thread utilization for K=1152 (5 active threads, 251 idle)
- Dequant + F16 GEMV has high parallelism in both stages

**Why not TVM/compiler approach?**
- TVM requires large WASM runtime (~50MB)
- Hand-tuned WGSL is more debuggable
- Profile-based selection is simpler than compiler IR
- Can always add compiler later if needed

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

