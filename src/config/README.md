# DOPPLER Config System

This directory contains the configuration system for DOPPLER inference.

## Directory Structure

```
src/config/
├── README.md                    # This file
├── runtime.js                   # Runtime config get/set API
├── kernel-path-loader.js        # Kernel path registry and resolution
├── schema/                      # JSON schemas and defaults
│   ├── runtime.schema.js        # Runtime config schema
│   └── kernel-path.schema.js    # Kernel path schema
└── presets/
    ├── runtime/                 # Runtime presets (default, debug, bench)
    ├── models/                  # Model family presets (gemma2, gemma3, llama3)
    └── kernel-paths/            # Explicit kernel pipeline definitions
```

## Concepts

### Model Presets vs Kernel Paths

**Model presets** (`presets/models/*.json`) define architecture-level properties:
- Attention: sliding window, softcapping, query-key norm
- Normalization: RMSNorm epsilon, weight offset
- FFN: activation function (gelu, silu)
- Detection patterns for auto-identification

**Kernel paths** (`presets/kernel-paths/*.json`) define which WGSL kernel to use for each operation:
- Decode steps: matmul variant, attention variant, activation kernel
- Prefill steps: batched variants
- Pre/post layer: embedding gather, final norm, lm_head

### Naming Convention

Kernel paths are named by **model family**, not size:

```
gemma2-q4k-dequant-f16a
│      │   │       └── Activation dtype (f16 or f32)
│      │   └────────── Kernel strategy (dequant or fused)
│      └────────────── Weight quantization (f16, q4k)
└───────────────────── Model family (gemma2, gemma3, llama3)
```

**Why family-level?** All sizes in a family (2B, 9B, 27B) share the same architecture.
Dimensions come from the model manifest at runtime.

## Kernel Path Registry

| ID | Default For | Description |
|----|-------------|-------------|
| `gemma2-f16-f16a` | Gemma 2 F16 | F16 weights, F16 activations |
| `gemma2-f16-f32a` | Gemma 2 F16 (no shader-f16) | F16 weights, F32 activations |
| `gemma2-q4k-dequant-f16a` | Gemma 2 Q4K | Q4K dequant path (safe) |
| `gemma2-q4k-fused-f16a` | (opt-in) | Q4K fused path (experimental) |
| `gemma2-q4k-fused-f32a` | Gemma 2 Q4K + F32 compute | Q4K fused with F32 |
| `gemma3-f16-f16a` | Gemma 3 F16 | F16, no softcapping |
| `gemma3-q4k-dequant-f16a` | Gemma 3 Q4K | Q4K dequant, no softcapping |

### Default Selection Logic

At conversion time, `src/converter/manifest-inference.js` selects the default kernel path:

```
Model preset kernelPaths[weightQuant][computeDtype] → manifest.inference.defaultKernelPath
```

Example for Gemma 2 Q4K with F16 compute:
```
gemma2.json → kernelPaths.q4k.f16 → "gemma2-q4k-dequant-f16a"
```

## Runtime Config

Runtime presets control logging, tracing, benchmarking, and inference parameters.

| Preset | Purpose |
|--------|---------|
| `default` | Production settings |
| `debug` | Verbose logging, tracing enabled |
| `bench` | Benchmarking settings (deterministic sampling) |

### Override Hierarchy

```
defaults → preset → config file → inline JSON → CLI flags
```

## Adding a New Model Family

1. **Create model preset** in `presets/models/{family}.json`:
   - Define architecture properties
   - Add `kernelPaths` mapping

2. **Create kernel paths** in `presets/kernel-paths/{family}-*.json`:
   - Copy from similar family (gemma2 → gemma3)
   - Adjust constants (softcapping, etc.)

3. **Register kernel paths** in `kernel-path-loader.js`:
   - Import JSON files
   - Add to `KERNEL_PATH_REGISTRY`

4. **Test**:
   - Convert model: `npx tsx src/converter/node-converter.js ...`
   - Verify manifest has `defaultKernelPath`
   - Run inference: `npm run debug -- -m MODEL`

## MoE Models (GPT-OSS, Mixtral)

MoE models use **runtime kernel selection** instead of explicit kernel paths.
The `moe.js` and `scatter_add.js` kernels are selected dynamically based on:
- Number of experts
- Tokens per expert
- GPU capabilities
Runtime selection should use rule maps (not ad-hoc if/ternary) so variant
choices stay auditable and consistent with the style guides.

MoE kernel path configs are not yet implemented.

## Kernel Audit

To verify kernel references:

```bash
# Kernels referenced in kernel-path configs
for f in src/config/presets/kernel-paths/*.json; do
  jq -r '.. | objects | select(.kernel) | .kernel' "$f"
done | sort -u

# Available kernel files
ls src/gpu/kernels/*.wgsl | xargs -I {} basename {}
```
