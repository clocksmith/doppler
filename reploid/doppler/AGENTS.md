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

### CLI Setup
Add alias to `~/.zshrc`: `alias doppler='npm run doppler --'`

### CLI Commands (3 modes)
```bash
# TEST - Correctness (does it work?)
doppler test                    # Quick kernel tests
doppler test --inference        # Model loads + generates (smoke test)
doppler test --full             # Full test suite

# BENCH - Performance (how fast?)
doppler bench                   # Full inference benchmark (tok/s)
doppler bench --kernels         # Kernel microbenchmarks
doppler bench --runs 3          # Multiple runs

# DEBUG - Debugging (why is it broken?)
doppler debug                   # Debug with kernel trace enabled
doppler debug --break           # Stop on first anomaly (NaN/explosion)
doppler debug --trace-layers 0,5  # Trace specific layers

# CPU tests
npm run test:vitest             # CPU unit tests
```

### Guardrails
- All GPU operations must handle device loss gracefully
- Validate tensor shapes at kernel boundaries
- Use BF16 for weights, F32 for activations
- Test with multiple quantization levels (Q4, Q8, F16)

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

