## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Write TypeScript for the WebGPU inference engine running in the browser.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent (separate repo).

### Directory Structure
```
doppler/
├── README.md                 ← Overview and quick start
├── AGENTS.md                 ← Agent instructions (this file)
├── CLAUDE.md → AGENTS.md
├── EMOJI.md                  ← Approved Unicode symbols
├── src/                      ← Source code
│   ├── inference/            ← Pipeline, attention, FFN
│   ├── gpu/                  ← WebGPU device, buffer pools
│   ├── storage/              ← OPFS shard manager
│   ├── loader/               ← GGUF parsing, RDRR manifest
│   ├── browser/              ← Browser-specific code
│   ├── memory/               ← Memory management
│   ├── debug/                ← Logging and tracing
│   └── ...                   ← Other source modules
├── app/                      ← Demo UI application
├── cli/                      ← CLI entry point
├── kernel-tests/             ← GPU kernel validation
├── tools/                    ← CLI utilities
└── docs/                     ← Documentation
```

### Before Starting
- Read `docs/ARCHITECTURE.md` for system overview
- Read `docs/spec/RDRR_FORMAT.md` for model format specification
- Review `src/inference/` for pipeline implementation

### Style Guides
- [Coding Guide](docs/style/CODING_GUIDE.md) - Architecture, file organization, naming
- [TypeScript Style Guide](docs/style/TYPESCRIPT_STYLE_GUIDE.md) - Config-as-code, kernel wrappers
- [WGSL Style Guide](docs/style/WGSL_STYLE_GUIDE.md) - Shader structure, constants vs uniforms

### Key Paths
- `src/inference/` - Pipeline, attention, FFN, embeddings
- `src/gpu/` - WebGPU device management, buffer pools
- `src/storage/` - OPFS shard manager, model loading
- `src/loader/` - GGUF parsing, .rdrr manifest handling
- `kernel-tests/` - GPU kernel validation tests
- `app/` - Demo UI application
- `tools/` - CLI utilities (convert, serve, debug)

### Architecture
```
GGUF/RDRR -> Loader -> ShardManager -> Pipeline -> GPU Kernels -> Output
                           |
                         OPFS (cached weights)
```

### Key Concepts
- **RDRR:** Recursive DOPPLER Runtime Registry - sharded model format
- **Pipeline:** Orchestrates prefill and decode passes through transformer layers
- **Kernels:** Custom WGSL shaders for RMSNorm, attention, FFN operations
- **OPFS:** Origin Private File System for persistent model storage

### CLI Commands (5 scripts)

| Script | Purpose | Page Launched |
|--------|---------|---------------|
| `npm start` | Dev server | `http://localhost:8080/` (index.html) |
| `npm test` | Kernel correctness | `/doppler/kernel-tests/browser/index.html` |
| `npm run bench` | Performance | `/doppler/tests/test-inference.html` |
| `npm run debug` | Debugging | `/doppler/tests/test-inference.html?debug=1` |
| `npm run build` | Compile TS | (no browser) |

```bash
# START - Dev server
npm start                       # Serve at localhost:8080
npx tsx serve.ts --port 3000    # Custom port (use npx directly)
npx tsx serve.ts --open         # Auto-open browser

# TEST - Correctness (does it work?)
npm test                        # Quick kernel tests
npm test -- --inference         # Model loads + generates (smoke test)
npm test -- --full              # Full test suite
npm test -- --filter matmul     # Filter to specific kernel

# BENCH - Performance (how fast?)
npm run bench                   # Full inference benchmark (tok/s)
npm run bench -- --kernels      # Kernel microbenchmarks
npm run bench -- --runs 3       # Multiple runs

# DEBUG - Debugging (why is it broken?)
npm run debug                   # Debug with all trace categories enabled
npm run debug -- --break        # Stop on first anomaly (NaN/explosion)
npm run debug -- --trace kernels,attn  # Trace specific categories
npm run debug -- --trace all,-buffers  # All except expensive buffer stats
npm run debug -- --trace-layers 0,5   # Filter to specific layers

# WARM MODE - Fast iteration (skip model reload)
# Use when debugging inference, not model loading itself.
# First run loads model, subsequent runs reuse it in GPU RAM.

# Option 1: Use Chrome with CDP (recommended)
# Terminal 1: Start Chrome with remote debugging
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --enable-unsafe-webgpu

# Terminal 2: First run loads model, keeps browser open
npm run debug -- --warm

# Terminal 2: Subsequent runs skip loading
npm run debug -- --skip-load    # Reuses pipeline from window.pipeline

# Option 2: Manual browser reuse
npm run debug -- --headed       # Keep browser visible
# Then on next run, pipeline persists if using same browser via CDP
```

### Common CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--model, -m <name>` | Model to use | `gemma-3-1b-it-q4` |
| `--verbose, -v` | Verbose loader logs | off |
| `--trace [cats]` | Trace categories (all if no arg) | off (on for debug) |
| `--quiet, -q` | Suppress logs | off |
| `--trace-layers <n,n>` | Filter trace to specific layers | all |
| `--headed` | Show browser window | off (headless default) |
| `--warm` | Keep browser open with model loaded | off |
| `--skip-load` | Skip model loading, reuse existing pipeline | off |
| `--timeout <ms>` | Test timeout | 120000 |
| `--output, -o <file>` | Save JSON results | none |
| `--kernel-profile, -k` | Preset: fast/safe/debug/fused/apple | none |

### Log Levels
Control output verbosity:

| CLI Flag | URL Param | Shows |
|----------|-----------|-------|
| (default) | `?log=info` | Phase starts/ends, totals |
| `--verbose, -v` | `?log=verbose` | + Per-shard source, per-layer timing |
| `--debug` | `?log=debug` | Full debug output |
| `--quiet, -q` | `?log=silent` | Errors only |

### Trace Categories
Control what gets traced with modular categories:

| CLI | URL | Effect |
|-----|-----|--------|
| `--trace` | `?trace=all` | All trace categories |
| `--trace kernels,logits` | `?trace=kernels,logits` | Specific categories |
| `--trace all,-buffers` | `?trace=all,-buffers` | All except buffers |

**Categories:** `loader`, `kernels`, `logits`, `embed`, `attn`, `ffn`, `kv`, `sample`, `buffers`, `perf`

**Defaults by mode:**
- `test`, `bench`: trace off (clean output)
- `debug`: `--trace` enabled (all categories)

**Shard source logs (verbose+):**
```
[Loader] Shard 0: RAM (64.0 MB)
[Loader] Shard 1: OPFS (64.0 MB, 0.05s)
[Loader] Shard 2: network (64.0 MB, 0.31s @ 206.5 MB/s)
```

### Logging Convention

**IMPORTANT:** All library code MUST use the unified debug module instead of raw `console.*` calls.

```typescript
// Import the debug module
import { log, trace } from '../debug/index.js';

// DON'T - raw console calls bypass log level control
console.log('[Pipeline] Model loaded');           // ❌
console.warn('[Attention] Fallback to CPU');      // ❌

// DO - use the debug module
log.info('Pipeline', 'Model loaded');             // ✓
log.warn('Attention', 'Fallback to CPU');         // ✓
log.debug('Matmul', `M=${M}, N=${N}, K=${K}`);    // ✓

// For trace categories (only logs if category enabled)
trace.kernels(`matmul M=${M} N=${N}`);            // ✓
trace.attn(layerIdx, 'Using chunked decode');     // ✓
trace.loader('Shard 0 from OPFS');                // ✓
```

**Why this matters:**
- `setLogLevel('silent')` silences all output
- `setBenchmarkMode(true)` mutes logs during benchmarks
- `enableModules('Pipeline')` filters to specific modules
- `getLogHistory()` captures logs for debugging
- Consistent formatting with timestamps

**Exceptions (raw console OK):**
- CLI tools in `tools/` - direct terminal output
- Test files in `kernel-tests/` - test harness output
- One-time startup messages in `device.ts`

### Guardrails
- All GPU operations must handle device loss gracefully
- Validate tensor shapes at kernel boundaries
- Use BF16 for weights, F32 for activations
- Test with multiple quantization levels (Q4, Q8, F16)

### Kernel Tests

Available kernel test suites (`npm test -- --filter <name>`):

| Kernel | Description |
|--------|-------------|
| `matmul` | Matrix multiplication (f16/f32) |
| `matmul-q4k` | Q4_K quantized matmul |
| `matmul-q4k-large` | Large Q4_K matmul |
| `attention` | Flash attention variants |
| `rmsnorm` | RMS normalization |
| `softmax` | Softmax with online normalization |
| `rope` | Rotary position embeddings |
| `silu` | SiLU activation |
| `swiglu` | SwiGLU activation (gated) |
| `gather` | Embedding lookup |
| `scatter-add` | MoE output combination |
| `moe-gather` | Token gathering by expert |
| `residual` | Residual addition |
| `topk` | Top-K selection |
| `dequant` | Dequantization (shared memory) |
| `dequant-q6k` | Q6_K dequantization |
| `scale` | Tensor scaling |
| `sample` | GPU-side sampling |

### CLI Tools

**Model Converter** (`src/converter/node-converter.ts`):

```bash
# Convert GGUF/SafeTensors to RDRR format
npx tsx src/converter/node-converter.ts <input> <output> [options]
  --shard-size <MB>      Shard size in MB (default: 64)
  --quantize <type>      Override quantization (q4_k_m, q6_k, q8_0, f16)
  --quantize-embeddings  Also quantize embedding weights
  --test                 Create tiny test fixture

# Examples:
npx tsx src/converter/node-converter.ts model.gguf models/my-model
npx tsx src/converter/node-converter.ts model.gguf models/my-model --quantize q4_k_m
```

**Dev Server** (`serve.ts`):

```bash
npx tsx serve.ts [options]
  --port, -p <n>         Server port (default: 8080)
  --open                 Auto-open browser

# Or use npm start
npm start
npm start -- --port 3000
```

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

