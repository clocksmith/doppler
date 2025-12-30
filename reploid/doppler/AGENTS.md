## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Write TypeScript for the WebGPU inference engine running in the browser.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent (separate repo).

### Directory Structure
```
doppler/
├── README.md                 ← Simple links (you are here at root)
├── AGENTS.md                 ← Symlink to reploid/doppler/AGENTS.md
├── CLAUDE.md                 ← Symlink to reploid/doppler/AGENTS.md
└── reploid/                  ← Intentionally empty
    └── doppler/              ← Actual project contents
        └── (gpu/, inference/, storage/, etc.)
```

### Before Starting
- Read `reploid/doppler/docs/ARCHITECTURE.md` for system overview
- Read `reploid/doppler/docs/spec/RDRR_FORMAT.md` for model format specification
- Review `reploid/doppler/inference/` for pipeline implementation

### Key Paths (relative to `reploid/doppler/`)
- `inference/` - Pipeline, attention, FFN, embeddings
- `gpu/` - WebGPU device management, buffer pools
- `storage/` - OPFS shard manager, model loading
- `loader/` - GGUF parsing, .rdrr manifest handling
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
npm start -- --port 3000        # Custom port
npm start -- --open             # Auto-open browser

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
npm run debug -- --layers 0,5   # Filter to specific layers
```

### Common CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--model, -m <name>` | Model to use | `gemma-3-1b-it-q4` |
| `--verbose, -v` | Verbose loader logs | off |
| `--trace [cats]` | Trace categories (all if no arg) | off (on for debug) |
| `--quiet, -q` | Suppress logs | off |
| `--layers <n,n>` | Filter trace to specific layers | all |
| `--headed` | Show browser window | off (headless default) |
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

### Guardrails
- All GPU operations must handle device loss gracefully
- Validate tensor shapes at kernel boundaries
- Use BF16 for weights, F32 for activations
- Test with multiple quantization levels (Q4, Q8, F16)

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

