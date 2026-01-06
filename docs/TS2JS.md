# Doppler TypeScript → JavaScript Conversion Plan

## Goals and Constraints (Hot-Swap Focus)

- Runtime code ships as JS in both dev and production; hot-swaps target JS/WGSL/JSON artifacts, not TS sources.
- JS output must be human/agent-readable (no minify), preserve names, and include sourcemaps.
- Distributed hot-swaps are signed; local unsigned swaps are allowed only behind an explicit "local-only" flag.
- TypeScript remains the authoring language; the build emits JS + .d.ts.
- Scope: start with `src/` runtime code; defer `cli/`, `app/`, and `kernel-tests/` until the runtime path is stable.

## Signing and Trust Model (Recommended)

- **Local dev:** per-device key signs artifacts; unsigned loads are allowed only with an explicit "local-only" flag.
- **P2P distribution:** accept only signatures from an allowlist of trusted signer IDs; unknown/unsigned bundles are rejected by default.
- **Official releases:** signed by a shared signer service key; runtime trusts that signer by default.
- **Manifest:** includes artifact hashes + signer ID + signature so swaps are auditable and reproducible.

## Summary Statistics (src/ only)

Counts drift; refresh before execution.

| Metric | Count |
|--------|-------|
| Total .ts files (excluding .d.ts) | 168 |
| Lines of TypeScript | ~57,000 |
| Files with `import type` | 81 |
| Existing .d.ts files | 1 (chrome.d.ts) |

Refresh commands:
```bash
rg --files -g '*.ts' -g '!*.d.ts' src | wc -l
rg --files -g '*.ts' -g '!*.d.ts' src | xargs wc -l | tail -n 1
rg -l "import\\s+type" src -g '*.ts' -g '!*.d.ts' | wc -l
```

---

## TypeScript Feature Usage

These counts are illustrative; refresh with a TS analyzer before sizing work.

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

### TypeScript-Light Files (Easy to Emit)
These use only basic type annotations and should emit clean JS:

| Category | Lines | Example Files |
|----------|-------|---------------|
| GPU kernels | ~15k | `sample.ts`, `constants.ts` |
| Simple utilities | ~3k | `sampling.ts`, `tokenizer.ts` |
| Debug/logging | ~3k | `debug/index.ts` |
| **Subtotal** | **~21k** | ~29% of codebase |

### TypeScript-Heavy Files (Higher Risk for Hot-Swap)
These use generics, utility types, interface extension:

| Category | Lines | Reason |
|----------|-------|--------|
| Config schemas | ~3,700 | Nested Partial<T>, deep merging |
| Pipeline core | ~10k | Complex LayerContext, generics |
| Loader/converter | ~5k | Generic tensor types |
| Type definitions | ~580 | src/types/*.ts |
| **Subtotal** | **~19k** | ~26% of codebase |

### Mixed Files (Moderate Risk)
| Category | Lines |
|----------|-------|
| Inference pipeline | ~25k |
| GPU operations | ~7k |
| **Subtotal** | **~32k** (~45%) |

---

## Conversion Strategy (Chosen Path)

**Approach:** Keep TypeScript sources and emit readable JS + .d.ts artifacts for hot-swapping.

Current config already generates:
- `declaration: true` → .d.ts files
- `declarationMap: true` → source maps

Output requirements:
- Emit unminified ESM JS, preserve names, and include sourcemaps.
- JS/WGSL/JSON artifacts are the swap unit; TS stays authoring-only.
- Signed manifests are required for distributed swaps.

---

## Type Emission Strategy

- Keep types in TS sources; rely on `tsc --declaration` for .d.ts output.
- Use `export type` in barrels to avoid accidental runtime imports.
- Before converting any file to .d.ts-only, confirm it exports no runtime values (const/function/class/enum).
- Type-only modules remain .ts (no runtime exports) and emit .d.ts (e.g., `src/types/*.ts`, pipeline/format type files).

### Target .d.ts Structure:
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

## Conversion Plan: Runtime JS Artifacts (Hot-Swappable) + .d.ts

### Phase 0: Setup (Before Converting Any Files)

1. **Define runtime artifact policy:**
   - Emit unminified ESM JS, preserve names, include sourcemaps.
   - JS/WGSL/JSON outputs are the hot-swap units.

2. **Add signing + verification hooks:**
   - Signed manifest for distributed swaps.
   - Allow local unsigned swaps only behind an explicit "local-only" flag.
   - Default trust: per-device signer for local, shared signer for official, allowlist for P2P.

3. **Generate baseline .d.ts files (project-level):**
   ```bash
   tsc -p tsconfig.build.json --declaration --emitDeclarationOnly --outDir dist/types
   ```

4. **Create types barrel file:**
   ```typescript
   // src/types/index.ts - consolidate all exports
   export type * from './gpu.js';
   export type * from './inference.js';
   export type * from './model.js';
   ```
   - This relies on `moduleResolution: "bundler"` and `package.json` `exports/types` to map `.js` specifiers to `.d.ts` at type-check time.
   - If module resolution changes, switch to `.d.ts` specifiers or add a `typesVersions` map.

5. **Confirm tsconfig.json + tsconfig.build.json:**
   ```json
   {
     "allowJs": true,
     "checkJs": true,
     "declaration": true,
     "emitDeclarationOnly": false
   }
   ```
   - `tsconfig.build.json` is the build driver (extends `tsconfig.json`); use `tsc -p tsconfig.build.json` for declarations and build outputs.

---

## Phase 1: Types Directory (5 files, ~580 lines)
These are type-only modules; keep as .ts (no runtime exports):

Guard before treating a file as type-only:
```bash
rg -n "export\\s+(const|function|class|let|var|enum)" src/types
```

| File | Lines | Action |
|------|-------|--------|
| `src/types/chrome.d.ts` | 36 | Keep as .d.ts |
| `src/types/index.ts` | 3 | Keep as .ts (type-only) |
| `src/types/gpu.ts` | 184 | Keep as .ts (type-only) |
| `src/types/inference.ts` | 199 | Keep as .ts (type-only) |
| `src/types/model.ts` | 118 | Keep as .ts (type-only) |

---

## Phase 2: Config Schema + Defaults (24 files, ~3,700 lines)
These are runtime defaults + helpers. Keep runtime JS and emit .d.ts (no .d.ts-only).

| File | Lines | Action |
|------|-------|--------|
| `src/config/schema/index.ts` | 100 | → .js + .d.ts barrel |
| `src/config/schema/doppler.schema.ts` | 266 | → .js + .d.ts |
| `src/config/schema/manifest.schema.ts` | 343 | → .js + .d.ts |
| `src/config/schema/inference.schema.ts` | 290 | → .js + .d.ts |
| `src/config/schema/inference-defaults.schema.ts` | 180 | → .js + .d.ts |
| `src/config/schema/debug.schema.ts` | 232 | → .js + .d.ts |
| `src/config/schema/loading.schema.ts` | 150 | → .js + .d.ts |
| `src/config/schema/preset.schema.ts` | 200 | → .js + .d.ts |
| `src/config/schema/storage.schema.ts` | 120 | → .js + .d.ts |
| `src/config/schema/distribution.schema.ts` | 100 | → .js + .d.ts |
| `src/config/schema/kvcache.schema.ts` | 80 | → .js + .d.ts |
| `src/config/schema/moe.schema.ts` | 90 | → .js + .d.ts |
| `src/config/schema/buffer-pool.schema.ts` | 70 | → .js + .d.ts |
| `src/config/schema/gpu-cache.schema.ts` | 60 | → .js + .d.ts |
| `src/config/schema/tuner.schema.ts` | 80 | → .js + .d.ts |
| `src/config/schema/memory-limits.schema.ts` | 70 | → .js + .d.ts |
| `src/config/schema/bridge.schema.ts` | 100 | → .js + .d.ts |
| `src/config/schema/platform.schema.ts` | 50 | → .js + .d.ts |
| `src/config/schema/hotswap.schema.ts` | 52 | → .js + .d.ts |
| `src/config/schema/kernel-registry.schema.ts` | 80 | → .js + .d.ts |
| `src/config/schema/kernel-plan.schema.ts` | 77 | → .js + .d.ts |
| `src/config/schema/kernel-thresholds.schema.ts` | 263 | → .js + .d.ts |
| `src/config/schema/quantization-defaults.schema.ts` | 46 | → .js + .d.ts |
| `src/config/schema/conversion.schema.ts` | 100 | → .js + .d.ts |

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
| `src/gpu/tensor.ts` | 200 | Medium | → .js |
| `src/gpu/weight-buffer.ts` | 180 | Simple | → .js |
| `src/gpu/command-recorder.ts` | 400 | Medium | → .js |
| `src/gpu/kernel-tuner.ts` | 1,261 | Complex | → .js |
| `src/gpu/kernel-selector.ts` | 300 | Medium | → .js |
| `src/gpu/kernel-runtime.ts` | 200 | Simple | → .js |
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

## Phase 5: Loader Module (8 files, ~3,500 lines)

| File | Lines | Complexity | Action |
|------|-------|------------|--------|
| `src/loader/doppler-loader.ts` | 1,944 | Complex | → .js |
| `src/loader/loader-types.ts` | 150 | Medium | → .d.ts |
| `src/loader/quantization-constants.ts` | 23 | Simple | → .js |
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

| Action | Files | Notes |
|--------|-------|-------|
| JS runtime artifacts + .d.ts | majority | all runtime modules, including schema defaults |
| Type-only TS (emit .d.ts) | few | `src/types/*.ts`, pipeline/format type files |
| .d.ts only | minimal | `src/types/chrome.d.ts` |
| **Total (src/)** | 168 | ~57,000 lines |

---

## Scope Roadmap

- **Phase A (now):** `src/` only (runtime hot-swap surface).
- **Phase B:** `cli/` (config tooling + dev workflows) once runtime artifacts are stable.
- **Phase C:** `app/` (demo UI) after CLI is aligned with signed bundles.
- **Phase D:** `kernel-tests/` last; keep TS for test ergonomics unless runtime parity requires JS.

## .d.ts File Structure (Final)

```
src/types/
├── index.d.ts              # Master barrel
├── gpu.d.ts                # GPU types (184 lines)
├── inference.d.ts          # Inference types (199 lines)
├── model.d.ts              # Model types (118 lines)
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

## Project-Level Conversion Steps

1. **Build JS output for the target scope:**
   - Emit ESM JS, no minify, preserve names, include sourcemaps.
   - Keep module boundaries stable for hot-swap.

2. **Generate declarations:**
   ```bash
   tsc -p tsconfig.build.json --declaration --emitDeclarationOnly --outDir dist/types
   ```

3. **Update imports/barrels:**
   - Use `export type` in .d.ts barrels to avoid accidental value imports.
   - Ensure `.js` extensions in JS runtime imports.

4. **Verify:**
   ```bash
   tsc -p tsconfig.json --noEmit --allowJs --checkJs
   ```

---

## Verification

After conversion:
```bash
# Type check with .d.ts
tsc -p tsconfig.json --noEmit --allowJs --checkJs

# Build uses the build config
tsc -p tsconfig.build.json

# Runtime test
npm test
```

Also validate signed swap acceptance and local-only bypass in the test harness.

---

## Types Resolution Appendix (Exports Map)

Type barrels use `.js` specifiers but resolve to `.d.ts` via `package.json` exports:

```json
{
  "exports": {
    ".": {
      "types": "./dist/types/src/index.d.ts",
      "import": "./dist/src/index.js"
    }
  }
}
```

If the exports map changes or module resolution differs, update barrels to `.d.ts` specifiers or add a `typesVersions` map.

---

## Challenges

### 1. Config Schema Runtime Defaults
`src/config/schema/*.ts` exports runtime defaults and helpers (e.g., `createDopplerConfig`, `DEFAULT_*`). These must remain runtime JS (or TS with JS output), not .d.ts-only.

### 2. Hot-Swap Trust Model
Production swaps require signature verification with a trusted signer allowlist, while local-only overrides remain explicit and non-distributable.

### 3. Declaration Hygiene
Ensure type-only exports use `export type` and avoid value imports from .d.ts barrels.

### 4. Artifact Stability
Keep module boundaries and output naming stable (no minify, preserve names) so hot-swaps are reproducible.
