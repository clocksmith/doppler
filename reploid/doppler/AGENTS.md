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

### CLI Commands (4 modes)
```bash
npm start                       # Serve demo app

# TEST - Correctness (does it work?)
npm test                        # Quick kernel tests
npm test -- inference           # Model loads + generates (smoke test)
npm test -- all                 # Full test suite

# BENCH - Performance (how fast?)
npm run bench                   # Full inference benchmark (tok/s)
npm run bench -- kernels        # Kernel microbenchmarks
npm run bench -- --runs 3       # Multiple runs

# DEBUG - Debugging (why is it broken?)
npm run debug                   # Debug with kernel trace enabled
npm run debug -- --break        # Stop on first anomaly (NaN/explosion)
npm run debug -- --trace-layers 0,5  # Trace specific layers
```

### Log Levels
Control loader output verbosity with CLI flags or browser URL params:

| CLI Flag | URL Param | Level | Shows |
|----------|-----------|-------|-------|
| (default) | `?log=info` | info | Phase starts/ends, totals |
| `--verbose` | `?log=verbose` | verbose | + Per-shard source (RAM/OPFS/network), per-layer timing |
| `--trace` | `?log=trace` | trace | + Tensor shapes, dequant ops, buffer details |
| `--quiet` | `?log=silent` | silent | Errors only |

**Defaults by mode:**
- `test`, `bench`: info (clean output)
- `debug`: verbose (shows shard sources and layer timing)

**Shard source logs (verbose+):**
```
[Loader] Shard 0: RAM (64.0 MB)
[Loader] Shard 1: OPFS (64.0 MB, 0.05s)
[Loader]  Shard 2: network (64.0 MB, 0.31s @ 206.5 MB/s)
```

### Guardrails
- All GPU operations must handle device loss gracefully
- Validate tensor shapes at kernel boundaries
- Use BF16 for weights, F32 for activations
- Test with multiple quantization levels (Q4, Q8, F16)

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

