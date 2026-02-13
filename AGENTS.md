## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Write JavaScript (with `.d.ts`) for the WebGPU inference engine and its unified browser/CLI command surface.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent orchestration.

### ⚠️ MANDATORY: Read Style Guides First

1. **[General Style Guide](docs/style/general-style-guide.md)**
2. **[JavaScript Guide](docs/style/javascript-style-guide.md)**
3. **[WGSL Guide](docs/style/WGSL_style-guide.md)**
4. **[Command Interface Design Guide](docs/style/command-interface-design-guide.md)**

These guides define performance and architecture invariants. Do not bypass them.

### Directory Structure

```
doppler/
├── src/
│   ├── inference/        # Pipeline, attention, FFN, embeddings
│   ├── gpu/              # WebGPU device, buffer pools, kernels
│   ├── storage/          # OPFS shard manager, model loading
│   ├── loader/           # GGUF parsing, RDRR manifest
│   ├── config/           # Runtime/model schemas + presets
│   ├── tooling/          # Shared command contract + runners
│   ├── memory/           # Heap management, capability detection
│   └── debug/            # Logging and tracing
├── tools/                # CLI entry points and scripts
├── tests/                # Browser harnesses + kernel tests
├── demo/                 # Browser UI
└── docs/                 # Documentation
```

### Before Starting

- Read `docs/architecture.md` for system overview.
- Read `docs/formats.md` for RDRR format constraints.
- Review `src/inference/pipeline.js` for inference flow.
- Review `src/tooling/command-api.js` for command parity contract.

### Command Surfaces (1:1 Contract)

- Browser entry: `runBrowserCommand()` via `src/tooling/browser-command-runner.js`
- Node entry: `runNodeCommand()` via `src/tooling/node-command-runner.js`
- CLI entrypoint: `tools/doppler-cli.mjs`

Rules:
- New commands must be added to `src/tooling/command-api.js`.
- Command semantics must match on browser and CLI.
- Unsupported environment capabilities must fail fast (never silent fallback).

### Config System

Use runtime presets/config payloads, not ad-hoc per-field flags.

- Runtime presets: `src/config/presets/runtime/`
- Model presets: `src/config/presets/models/`
- Read tunables via `getRuntimeConfig()`; avoid hardcoded defaults in runtime paths.
- `runtime.shared.tooling.intent` is required for harnessed debug/bench/test flows.

### Conversion Triage Protocol (Required)

When a freshly converted model regresses, separate conversion integrity from runtime regressions before changing presets:

1. Verify source dtypes from checkpoint headers (`BF16`/`F16`/`F32` mix).
2. Verify converted manifest fields: `quantization`, `quantizationInfo`, `inference.defaultKernelPath`.
3. Verify shard integrity (sampled shard hashes must match manifest hashes).
4. Verify numeric sanity by sampling tensor values from source vs converted bytes.
5. Verify parsed layer pattern semantics from manifest (Gemma `every_n` is layer 0 + every N).

Do not claim a conversion bug unless steps 1-4 fail.
Do not claim a runtime bug unless steps 1-4 pass and runtime still diverges.

### Logging

Use debug module (`src/debug/index.js`), not raw `console.*` in runtime code.

Allowed direct console output:
- `tools/` entry points
- `tests/` harnesses
- `demo/` entry points
- one-time startup in `src/gpu/device.js`

### No Ad-Hoc Debug Logging

Do not add throwaway log statements.
Use:
1. existing trace categories,
2. config-driven probes,
3. permanent trace extensions.

### Guardrails

- Handle GPU device loss gracefully.
- Validate tensor shapes at kernel boundaries.
- Keep command/runtime behavior deterministic under fixed config.
- Do not ship surface-specific command behavior drift.

### Skills

- `doppler-debug`: debug inference issues.
- `doppler-bench`: run performance benchmarks.
- `doppler-convert`: convert models to RDRR.

See `docs/KERNEL_COMPATIBILITY.md` for kernel overrides and runtime modes.
