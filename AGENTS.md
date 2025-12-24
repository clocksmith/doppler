## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Write TypeScript for the WebGPU inference engine running in the browser.

### Before Starting
- Read `docs/ARCHITECTURE.md` for system overview
- Read `docs/spec/RDRR_FORMAT.md` for model format specification
- Review `inference/` for pipeline implementation

### Key Paths
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

### Testing
```bash
doppler test quick         # Fast kernel validation (default)
doppler test correctness   # Full kernel correctness tests
doppler test inference     # Model load + generate test
doppler bench inference    # Performance benchmarks
npm run test:vitest        # CPU unit tests
```

### Guardrails
- All GPU operations must handle device loss gracefully
- Validate tensor shapes at kernel boundaries
- Use BF16 for weights, F32 for activations
- Test with multiple quantization levels (Q4, Q8, F16)

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

