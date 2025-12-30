# DOPPLER Config System

Purpose: Runtime configuration, schemas, and kernel-path selection for inference.

## Scope

- Runtime config APIs, schemas, and preset loading.
- Model family presets and kernel path registries.

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

## End-to-End Config Domains (Config-First)

Treat configuration as four explicit domains:

1. Conversion config: artifact policy (`quantization`, sharding, manifest policy, output model ID/path).
2. Runtime config: execution policy (`shared`, `loading`, `inference`).
3. Benchmark shared contract: fairness/workload policy (prompt/template, sampling, seed, token budgets, warm/cold policy, run counts).
4. Benchmark engine overlay: engine-only knobs (Doppler decode cadence/kernel path vs vendor backend/session settings).

Do not mix benchmark fairness axes with engine internals in a single object.

## Target Preset Layout (Proposed)

Use this target layout for new presets and gradual migration:

```
src/config/presets/
├── conversion/                  # Conversion presets (artifact creation)
│   ├── models/                  # Family conversion defaults
│   └── variants/                # dtype/quantization variants (f16, f32a, q4k, ...)
├── runtime/                     # Runtime presets (execution only)
│   ├── intents/                 # verify / investigate / calibrate
│   ├── workloads/               # prefill/decode/sampling workload tuples
│   ├── overlays/                # doppler engine overlays (batch/readback/kernelPath)
│   └── diagnostics/             # trace/log/probe overlays
├── benchmarks/                  # Cross-engine benchmark contracts
│   ├── shared/                  # apples-to-apples workload definitions
│   └── engines/                 # engine overlays (doppler, tjs, ...)
├── models/                      # Manifest inference presets by model family
└── kernel-paths/                # Kernel path registry + path specs
```

## Migration Rules

- Keep existing preset paths stable while migrating; add aliases/compat docs first.
- Move presets by domain ownership, not by team or feature.
- Preserve command parity: browser and Node must consume the same shared benchmark contract.
- New benchmark tooling should accept: `sharedContract + engineOverlay`, never ad-hoc per-field flags as the primary API.
- Promote experimental presets to canonical only after correctness and performance gates are documented.

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

Source of truth for kernel-path IDs is `src/config/presets/kernel-paths/registry.json`.

This registry is the only place to define kernel-path identity, status, and compatibility.
`kernel-path-loader.js` resolves all IDs from this file at runtime.

### Path Status

- `canonical`: expected for conversion defaults and public use.
- `experimental`: active probes/tuning path, not implied as a default.
- `legacy`: compatibility-only entry that must resolve to a canonical target.

```bash
node -e "const fs = require('node:fs'); const json = JSON.parse(fs.readFileSync('src/config/presets/kernel-paths/registry.json', 'utf8')); console.log(json.entries.map((e) => e.id).join(', '));"
```

To inspect lifecycle history and alias chains:

```bash
node -e "const fs = require('node:fs'); const json = JSON.parse(fs.readFileSync('src/config/presets/kernel-paths/registry.json', 'utf8')); console.log(json.entries.filter((e) => e.status === 'legacy').map((e) => `${e.id} -> ${e.aliasOf}`).join('\n'));"
```

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
defaults → preset → config file → inline JSON
```

## Adding a New Model Family

1. **Create model preset** in `presets/models/{family}.json`:
   - Define architecture properties
   - Add `kernelPaths` mapping

2. **Create kernel paths** in `presets/kernel-paths/{family}-*.json`:
   - Copy from similar family (gemma2 → gemma3)
   - Adjust constants (softcapping, etc.)

3. **Register kernel paths** in `src/config/presets/kernel-paths/registry.json`:
  - Add a new entry to `src/config/presets/kernel-paths/registry.json`
  - Set `status` (`canonical`, `experimental`, or `legacy`)
  - Reference the JSON file for canonical paths, or `aliasOf` for compatibility-only aliases
  - Keep old IDs as `legacy` and point them to the replacement target until callers migrate

4. **Test**:
   - Convert model in the demo UI (Import → Convert)
   - Verify manifest has `defaultKernelPath`
   - Run inference via diagnostics UI or `tests/harness.html` (runtime config defines model)

## MoE Models (GPT-OSS, Mixtral)

MoE models use **runtime kernel selection** with explicit profile config:
- `src/config/kernels/moe/gpt-oss.paths.json` (profile rules)
- `src/rules/kernels/moe.rules.gptoss.json` (vendor quirks and constraints)

The `moe` runtime resolves kernels from rule maps and capability context
(`hasF16`, `hasSubgroups`, vendor), then validates shape policy before dispatch.
No silent fallback is allowed in GPT-OSS mode when required capabilities are absent.

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
