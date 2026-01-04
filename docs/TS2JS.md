# Doppler TypeScript → JavaScript Conversion Plan

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total .ts files | 164 |
| Lines of TypeScript | ~72,000 |
| Exported types/interfaces | 577 |
| Files with `import type` | 228 |
| Existing .d.ts files | 1 (chrome.d.ts) |

---

## TypeScript Feature Usage

| Feature | Count | Complexity |
|---------|-------|------------|
| Interfaces | 409 exports | Heavy in config/schema/ |
| Type aliases | 168 exports | Union types throughout |
| Generics | 566 uses | Widespread |
| Utility types (Partial, Record, etc.) | 287 uses | Config merging |
| Type guards/narrowing | 994 uses | Runtime type checks |
| Enums | 0 | Uses string literals instead |
| Decorators | 0 | Not used |
| Conditional types | 0 | Not used |

---

## Type Complexity Distribution

| Complexity | % of Types | Example |
|------------|------------|---------|
| Simple (1-5 props) | 40% | `SoftmaxOptions`, `BufferRequest` |
| Medium (6-15 props) | 50% | `InferenceConfig`, `KVCacheConfig` |
| Complex (16+ props) | 10% | `ManifestSchema`, `RuntimeConfigSchema` |

---

## What Could Be JavaScript

### TypeScript-Light Files (Easy to Convert)
These use only basic type annotations that could be JSDoc or stripped:

| Category | Lines | Example Files |
|----------|-------|---------------|
| GPU kernels | ~15k | `sample.ts`, `constants.ts` |
| Simple utilities | ~3k | `sampling.ts`, `tokenizer.ts` |
| Debug/logging | ~3k | `debug/index.ts` |
| **Subtotal** | **~21k** | ~29% of codebase |

### TypeScript-Heavy Files (Hard to Convert)
These use generics, utility types, interface extension:

| Category | Lines | Reason |
|----------|-------|--------|
| Config schemas | 3,155 | Nested Partial<T>, deep merging |
| Pipeline core | ~10k | Complex LayerContext, generics |
| Loader/converter | ~5k | Generic tensor types |
| Type definitions | 546 | src/types/*.ts |
| **Subtotal** | **~19k** | ~26% of codebase |

### Mixed Files (Moderate Effort)
| Category | Lines |
|----------|-------|
| Inference pipeline | ~25k |
| GPU operations | ~7k |
| **Subtotal** | **~32k** (~45%) |

---

## Conversion Strategies

### Option 1: Strip Types → Pure JavaScript
**Effort:** 577 type exports need .d.ts files
**Lines to convert:** ~72k
**Tooling:** `tsc --declaration` already outputs .d.ts

**Pros:**
- Smaller runtime bundle (no type erasure overhead in source)
- Works in browsers without build step

**Cons:**
- Lose IDE type-checking during development
- Must maintain .d.ts files separately

### Option 2: JSDoc Annotations
**Effort:** Convert inline types to JSDoc comments
**Lines to modify:** ~50k (function signatures, variables)

Example conversion:
```typescript
// BEFORE (TypeScript)
function sample(logits: Float32Array, opts: SamplingOptions): number

// AFTER (JSDoc + JavaScript)
/** @param {Float32Array} logits @param {SamplingOptions} opts @returns {number} */
function sample(logits, opts)
```

**Pros:**
- TypeScript can still check .js files with `checkJs: true` (already enabled!)
- No .d.ts maintenance
- Works in browsers without build

**Cons:**
- More verbose
- Some advanced types can't be expressed in JSDoc

### Option 3: Keep TypeScript, Publish .d.ts
**Effort:** Already supported by current tsconfig
**Lines to modify:** 0

Current config already generates:
- `declaration: true` → .d.ts files
- `declarationMap: true` → source maps

**Pros:**
- Best DX (full IDE support)
- Consumers get types via .d.ts

**Cons:**
- Requires build step
- TypeScript in source

---

## Type Separation Feasibility

### Current State
- Types scattered across 164 files
- Heavy `import type` coupling (228 files)
- Config schemas tightly bound to defaults

### To Separate Types into .d.ts:

1. **Move 577 exports to .d.ts files** (~3k lines)
2. **Create type barrel file** (`src/types/index.d.ts`)
3. **Update 228 import statements** to reference .d.ts

### Estimated .d.ts Structure:
```
src/types/
├── index.d.ts          # Main barrel export
├── gpu.d.ts            # GPU kernel types
├── inference.d.ts      # Pipeline types
├── model.d.ts          # Model config types
├── config.d.ts         # Runtime config types
└── formats.d.ts        # GGUF/RDRR types
```

### Line Estimates for .d.ts Files:
| File | Lines |
|------|-------|
| gpu.d.ts | ~400 |
| inference.d.ts | ~600 |
| model.d.ts | ~300 |
| config.d.ts | ~800 |
| formats.d.ts | ~500 |
| **Total** | ~2,600 lines |

---

## Conversion Plan: TypeScript → JavaScript + .d.ts

### Phase 0: Setup (Before Converting Any Files)

1. **Generate baseline .d.ts files:**
   ```bash
   tsc --declaration --emitDeclarationOnly --outDir dist/types
   ```

2. **Create types barrel file:**
   ```typescript
   // src/types/index.d.ts - consolidate all exports
   export * from './gpu.js';
   export * from './inference.js';
   export * from './model.js';
   ```

3. **Update tsconfig.json:**
   ```json
   {
     "allowJs": true,
     "checkJs": true,
     "declaration": true,
     "emitDeclarationOnly": false
   }
   ```

---

## Phase 1: Types Directory (4 files, 550 lines)
These stay as .ts or become pure .d.ts:

| File | Lines | Action |
|------|-------|--------|
| `src/types/chrome.d.ts` | 50 | Keep as .d.ts |
| `src/types/gpu.ts` | 185 | → `gpu.d.ts` |
| `src/types/inference.ts` | 209 | → `inference.d.ts` |
| `src/types/model.ts` | 119 | → `model.d.ts` |

---

## Phase 2: Config Schema (20 files, 3,155 lines)
These are type-heavy. Convert to .d.ts only:

| File | Lines | Action |
|------|-------|--------|
| `src/config/schema/index.ts` | 100 | → .d.ts barrel |
| `src/config/schema/doppler.schema.ts` | 266 | → .d.ts |
| `src/config/schema/manifest.schema.ts` | 343 | → .d.ts |
| `src/config/schema/inference.schema.ts` | 290 | → .d.ts |
| `src/config/schema/inference-defaults.schema.ts` | 180 | → .d.ts |
| `src/config/schema/debug.schema.ts` | 232 | → .d.ts |
| `src/config/schema/loading.schema.ts` | 150 | → .d.ts |
| `src/config/schema/preset.schema.ts` | 200 | → .d.ts |
| `src/config/schema/storage.schema.ts` | 120 | → .d.ts |
| `src/config/schema/distribution.schema.ts` | 100 | → .d.ts |
| `src/config/schema/kvcache.schema.ts` | 80 | → .d.ts |
| `src/config/schema/moe.schema.ts` | 90 | → .d.ts |
| `src/config/schema/buffer-pool.schema.ts` | 70 | → .d.ts |
| `src/config/schema/gpu-cache.schema.ts` | 60 | → .d.ts |
| `src/config/schema/tuner.schema.ts` | 80 | → .d.ts |
| `src/config/schema/memory-limits.schema.ts` | 70 | → .d.ts |
| `src/config/schema/bridge.schema.ts` | 100 | → .d.ts |
| `src/config/schema/platform.schema.ts` | 50 | → .d.ts |
| `src/config/schema/kernel-registry.schema.ts` | 80 | → .d.ts |
| `src/config/schema/conversion.schema.ts` | 100 | → .d.ts |

**Config runtime files (convert to .js):**
| File | Lines | Action |
|------|-------|--------|
| `src/config/index.ts` | 50 | → .js |
| `src/config/loader.ts` | 548 | → .js |
| `src/config/runtime.ts` | 57 | → .js |

---

## Phase 3: GPU Module (42 files, ~15,000 lines)

### GPU Core (17 files)
| File | Lines | Complexity | Action |
|------|-------|------------|--------|
| `src/gpu/device.ts` | 600 | Medium | → .js |
| `src/gpu/buffer-pool.ts` | 500 | Medium | → .js |
| `src/gpu/buffer-dtypes.ts` | 100 | Simple | → .js |
| `src/gpu/command-recorder.ts` | 400 | Medium | → .js |
| `src/gpu/kernel-tuner.ts` | 1,261 | Complex | → .js |
| `src/gpu/kernel-selector.ts` | 300 | Medium | → .js |
| `src/gpu/kernel-runtime.ts` | 200 | Simple | → .js |
| `src/gpu/kernel-hints.ts` | 150 | Simple | → .js |
| `src/gpu/kernel-benchmark.ts` | 250 | Medium | → .js |
| `src/gpu/kernel-selection-cache.ts` | 100 | Simple | → .js |
| `src/gpu/uniform-cache.ts` | 200 | Simple | → .js |
| `src/gpu/profiler.ts` | 300 | Medium | → .js |
| `src/gpu/perf-profiler.ts` | 250 | Medium | → .js |
| `src/gpu/perf-guards.ts` | 100 | Simple | → .js |
| `src/gpu/submit-tracker.ts` | 150 | Simple | → .js |
| `src/gpu/partitioned-buffer-pool.ts` | 200 | Medium | → .js |
| `src/gpu/multi-model-recorder.ts` | 150 | Simple | → .js |

### GPU Kernels (25 files)
| File | Lines | Complexity | Action |
|------|-------|------------|--------|
| `src/gpu/kernels/index.ts` | 50 | Simple | → .js |
| `src/gpu/kernels/types.ts` | 22 | Simple | → .d.ts |
| `src/gpu/kernels/constants.ts` | 300 | Simple | → .js |
| `src/gpu/kernels/utils.ts` | 1,244 | Complex | → .js |
| `src/gpu/kernels/kernel-base.ts` | 150 | Medium | → .js |
| `src/gpu/kernels/dispatch.ts` | 200 | Medium | → .js |
| `src/gpu/kernels/matmul.ts` | 800 | Complex | → .js |
| `src/gpu/kernels/attention.ts` | 700 | Complex | → .js |
| `src/gpu/kernels/rmsnorm.ts` | 300 | Medium | → .js |
| `src/gpu/kernels/rope.ts` | 400 | Medium | → .js |
| `src/gpu/kernels/softmax.ts` | 250 | Medium | → .js |
| `src/gpu/kernels/silu.ts` | 150 | Simple | → .js |
| `src/gpu/kernels/gelu.ts` | 150 | Simple | → .js |
| `src/gpu/kernels/gather.ts` | 300 | Medium | → .js |
| `src/gpu/kernels/residual.ts` | 200 | Simple | → .js |
| `src/gpu/kernels/scale.ts` | 150 | Simple | → .js |
| `src/gpu/kernels/cast.ts` | 200 | Simple | → .js |
| `src/gpu/kernels/dequant.ts` | 500 | Medium | → .js |
| `src/gpu/kernels/sample.ts` | 500 | Medium | → .js |
| `src/gpu/kernels/moe.ts` | 400 | Medium | → .js |
| `src/gpu/kernels/split_qkv.ts` | 250 | Medium | → .js |
| `src/gpu/kernels/check-stop.ts` | 200 | Simple | → .js |
| `src/gpu/kernels/fused_ffn.ts` | 400 | Medium | → .js |
| `src/gpu/kernels/fused_matmul_residual.ts` | 300 | Medium | → .js |
| `src/gpu/kernels/fused_matmul_rmsnorm.ts` | 300 | Medium | → .js |

---

## Phase 4: Inference Module (26 files, ~18,000 lines)

### Inference Core (12 files)
| File | Lines | Complexity | Action |
|------|-------|------------|--------|
| `src/inference/pipeline.ts` | 1,868 | Complex | → .js |
| `src/inference/kv-cache.ts` | 1,044 | Complex | → .js |
| `src/inference/tokenizer.ts` | 1,512 | Medium | → .js |
| `src/inference/decode-buffers.ts` | 300 | Medium | → .js |
| `src/inference/expert-router.ts` | 400 | Medium | → .js |
| `src/inference/moe-router.ts` | 350 | Medium | → .js |
| `src/inference/functiongemma.ts` | 890 | Medium | → .js |
| `src/inference/speculative.ts` | 500 | Complex | → .js |
| `src/inference/test-harness.ts` | 600 | Medium | → .js |
| `src/inference/multi-pipeline-pool.ts` | 400 | Medium | → .js |
| `src/inference/multi-model-network.ts` | 350 | Medium | → .js |
| `src/inference/network-evolution.ts` | 300 | Medium | → .js |

### Inference Pipeline (16 files)
| File | Lines | Complexity | Action |
|------|-------|------------|--------|
| `src/inference/pipeline/types.ts` | 358 | Complex | → .d.ts |
| `src/inference/pipeline/buffer-types.ts` | 8 | Simple | → .d.ts |
| `src/inference/pipeline/lora-types.ts` | 50 | Simple | → .d.ts |
| `src/inference/pipeline/layer.ts` | 1,746 | Complex | → .js |
| `src/inference/pipeline/attention.ts` | 927 | Complex | → .js |
| `src/inference/pipeline/init.ts` | 829 | Complex | → .js |
| `src/inference/pipeline/debug-utils.ts` | 805 | Medium | → .js |
| `src/inference/pipeline/moe-impl.ts` | 716 | Complex | → .js |
| `src/inference/pipeline/config.ts` | 543 | Medium | → .js |
| `src/inference/pipeline/kernel-trace.ts` | 529 | Medium | → .js |
| `src/inference/pipeline/embed.ts` | 400 | Medium | → .js |
| `src/inference/pipeline/logits.ts` | 350 | Medium | → .js |
| `src/inference/pipeline/sampling.ts` | 300 | Medium | → .js |
| `src/inference/pipeline/weights.ts` | 400 | Medium | → .js |
| `src/inference/pipeline/layer-plan.ts` | 250 | Medium | → .js |
| `src/inference/pipeline/probes.ts` | 200 | Simple | → .js |
| `src/inference/pipeline/lora-apply.ts` | 300 | Medium | → .js |
| `src/inference/pipeline/lora.ts` | 250 | Medium | → .js |

---

## Phase 5: Loader Module (7 files, ~4,500 lines)

| File | Lines | Complexity | Action |
|------|-------|------------|--------|
| `src/loader/doppler-loader.ts` | 1,944 | Complex | → .js |
| `src/loader/loader-types.ts` | 150 | Medium | → .d.ts |
| `src/loader/weights.ts` | 500 | Medium | → .js |
| `src/loader/shard-cache.ts` | 400 | Medium | → .js |
| `src/loader/expert-cache.ts` | 500 | Medium | → .js |
| `src/loader/dtype-utils.ts` | 200 | Simple | → .js |
| `src/loader/multi-model-loader.ts` | 350 | Medium | → .js |

---

## Phase 6: Storage Module (7 files, ~3,000 lines)

| File | Lines | Complexity | Action |
|------|-------|------------|--------|
| `src/storage/shard-manager.ts` | 816 | Complex | → .js |
| `src/storage/downloader.ts` | 740 | Medium | → .js |
| `src/storage/download-types.ts` | 100 | Simple | → .d.ts |
| `src/storage/rdrr-format.ts` | 400 | Medium | → .js |
| `src/storage/quota.ts` | 200 | Simple | → .js |
| `src/storage/preflight.ts` | 300 | Medium | → .js |
| `src/storage/quickstart-downloader.ts` | 250 | Medium | → .js |

---

## Phase 7: Formats Module (18 files, ~3,500 lines)

| File | Lines | Complexity | Action |
|------|-------|------------|--------|
| `src/formats/index.ts` | 30 | Simple | → .js |
| `src/formats/gguf.ts` | 100 | Simple | → .js |
| `src/formats/gguf/index.ts` | 20 | Simple | → .js |
| `src/formats/gguf/parser.ts` | 600 | Medium | → .js |
| `src/formats/gguf/types.ts` | 508 | Complex | → .d.ts |
| `src/formats/safetensors.ts` | 50 | Simple | → .js |
| `src/formats/safetensors/index.ts` | 20 | Simple | → .js |
| `src/formats/safetensors/parser.ts` | 400 | Medium | → .js |
| `src/formats/safetensors/types.ts` | 150 | Medium | → .d.ts |
| `src/formats/tokenizer.ts` | 100 | Simple | → .js |
| `src/formats/tokenizer/index.ts` | 20 | Simple | → .js |
| `src/formats/tokenizer/types.ts` | 50 | Simple | → .d.ts |
| `src/formats/rdrr/index.ts` | 30 | Simple | → .js |
| `src/formats/rdrr/types.ts` | 200 | Medium | → .d.ts |
| `src/formats/rdrr/parsing.ts` | 300 | Medium | → .js |
| `src/formats/rdrr/manifest.ts` | 250 | Medium | → .js |
| `src/formats/rdrr/validation.ts` | 200 | Medium | → .js |
| `src/formats/rdrr/groups.ts` | 150 | Simple | → .js |
| `src/formats/rdrr/classification.ts` | 150 | Simple | → .js |

---

## Phase 8: Remaining Modules

### Memory (4 files)
| File | Lines | Action |
|------|-------|--------|
| `src/memory/heap-manager.ts` | 300 | → .js |
| `src/memory/capability.ts` | 200 | → .js |
| `src/memory/unified-detect.ts` | 250 | → .js |
| `src/memory/address-table.ts` | 150 | → .js |

### Debug (3 files)
| File | Lines | Action |
|------|-------|--------|
| `src/debug/index.ts` | 1,248 | → .js |
| `src/debug/tensor.ts` | 400 | → .js |
| `src/debug/diagnose-kernels.ts` | 300 | → .js |

### Converter (8 files)
| File | Lines | Action |
|------|-------|--------|
| `src/converter/index.ts` | 30 | → .js |
| `src/converter/core.ts` | 800 | → .js |
| `src/converter/writer.ts` | 1,082 | → .js |
| `src/converter/node-converter.ts` | 1,024 | → .js |
| `src/converter/quantizer.ts` | 400 | → .js |
| `src/converter/shard-packer.ts` | 300 | → .js |
| `src/converter/test-model.ts` | 200 | → .js |
| `src/converter/io/node.ts` | 250 | → .js |

### Adapters (5 files)
| File | Lines | Action |
|------|-------|--------|
| `src/adapters/index.ts` | 20 | → .js |
| `src/adapters/adapter-manager.ts` | 708 | → .js |
| `src/adapters/adapter-manifest.ts` | 200 | → .js |
| `src/adapters/adapter-registry.ts` | 300 | → .js |
| `src/adapters/lora-loader.ts` | 400 | → .js |

### Bridge (5 files)
| File | Lines | Action |
|------|-------|--------|
| `src/bridge/index.ts` | 20 | → .js |
| `src/bridge/protocol.ts` | 150 | → .js |
| `src/bridge/extension-client.ts` | 300 | → .js |
| `src/bridge/extension/background.ts` | 200 | → .js |
| `src/bridge/native/native-host.ts` | 400 | → .js |

### Browser (6 files)
| File | Lines | Action |
|------|-------|--------|
| `src/browser/browser-converter.ts` | 300 | → .js |
| `src/browser/file-picker.ts` | 150 | → .js |
| `src/browser/gguf-importer.ts` | 200 | → .js |
| `src/browser/gguf-parser-browser.ts` | 250 | → .js |
| `src/browser/safetensors-parser-browser.ts` | 200 | → .js |
| `src/browser/shard-io-browser.ts` | 150 | → .js |

### Root
| File | Lines | Action |
|------|-------|--------|
| `src/index.ts` | 50 | → .js |

---

## Summary by Action

| Action | Files | Lines |
|--------|-------|-------|
| Convert to .js | 130 | ~62,000 |
| Convert to .d.ts only | 34 | ~10,000 |
| **Total** | 164 | ~72,000 |

---

## .d.ts File Structure (Final)

```
src/types/
├── index.d.ts              # Master barrel
├── gpu.d.ts                # GPU types (185 lines)
├── inference.d.ts          # Inference types (209 lines)
├── model.d.ts              # Model types (119 lines)
├── config.d.ts             # All config schemas (~2,500 lines)
├── formats.d.ts            # GGUF/RDRR/SafeTensors types (~800 lines)
├── pipeline.d.ts           # Pipeline internal types (~400 lines)
├── loader.d.ts             # Loader types (~150 lines)
└── storage.d.ts            # Storage types (~100 lines)

Total: ~4,500 lines of .d.ts
```

---

## Execution Order

1. **Generate .d.ts baseline** (Phase 0)
2. **Convert types/** (Phase 1) - establishes type foundation
3. **Convert config/schema/** (Phase 2) - needed by everything
4. **Convert gpu/kernels/** (Phase 3) - self-contained
5. **Convert gpu/** core (Phase 3) - depends on kernels
6. **Convert formats/** (Phase 7) - standalone
7. **Convert loader/** (Phase 5) - depends on formats
8. **Convert storage/** (Phase 6) - depends on loader
9. **Convert inference/pipeline/** (Phase 4) - depends on gpu, loader
10. **Convert inference/** core (Phase 4) - depends on pipeline
11. **Convert remaining** (Phase 8) - all utilities

---

## Per-File Conversion Steps

For each .ts file:

1. **Strip types:**
   ```bash
   npx esbuild file.ts --outfile=file.js --format=esm
   ```

2. **Generate declaration:**
   ```bash
   npx tsc file.ts --declaration --emitDeclarationOnly
   ```

3. **Update imports:**
   - `import type { X }` → remove or keep for JSDoc
   - Ensure `.js` extensions

4. **Verify:**
   ```bash
   npx tsc --noEmit file.js
   ```

---

## Verification

After conversion:
```bash
# Type check with .d.ts
tsc --noEmit --allowJs --checkJs

# Runtime test
npm test
```

---

## Challenges

### 1. Config Schema Merging (3,155 lines)
`src/config/schema/*.ts` uses `Partial<T>` heavily for deep merging:
```typescript
export function createDopplerConfig(overrides?: Partial<DopplerConfigSchema>): DopplerConfigSchema
```
In JS, this loses type safety. Options:
- Keep schema files as .ts (hybrid approach)
- Use JSDoc `@typedef` for complex types

### 2. Interface Extension (GPU Kernels)
```typescript
export interface DequantOptions extends OutputBufferOptions, OutputOffsetOptions, OutputDtypeOptions
```
JSDoc equivalent is verbose:
```javascript
/** @typedef {OutputBufferOptions & OutputOffsetOptions & OutputDtypeOptions & {...}} DequantOptions */
```

### 3. Generic Functions
```typescript
async downloadShards<T extends ArrayBuffer | Blob>(...)
```
JSDoc generics are limited - may need runtime assertions.
