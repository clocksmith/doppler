## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Write JavaScript (with `.d.ts`) for the WebGPU inference engine and its unified browser/CLI command surface.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent orchestration.

### MANDATORY: Read Style Guides First

1. **[General Style Guide](docs/style/general-style-guide.md)**
2. **[JavaScript Guide](docs/style/javascript-style-guide.md)**
3. **[WGSL Guide](docs/style/wgsl-style-guide.md)**
4. **[Command Interface Design Guide](docs/style/command-interface-design-guide.md)**

These guides define performance and architecture invariants. Do not bypass them.

### Directory Structure

```
doppler/
├── src/
│   ├── adapters/         # External integration adapters
│   ├── bridge/           # Browser/Node bridge layer
│   ├── browser/          # Browser-specific entry points
│   ├── client/           # Client API surface
│   ├── config/           # Runtime/model schemas + presets
│   ├── converter/        # SafeTensors/GGUF → RDRR conversion
│   ├── debug/            # Logging and tracing
│   ├── diffusion/        # Image diffusion pipeline surface
│   ├── distribution/     # P2P shard transport (experimental)
│   ├── errors/           # Error types and handling
│   ├── formats/          # Format parsing utilities
│   ├── gpu/              # WebGPU device, buffer pools, kernels
│   ├── hotswap/          # Live model/adapter swap (experimental)
│   ├── inference/        # Pipeline, attention, FFN, embeddings
│   ├── generation/       # Text pipeline surface
│   ├── loader/           # GGUF parsing, RDRR manifest
│   ├── memory/           # Heap management, capability detection
│   ├── rules/            # Validation rules (kernel path, etc.)
│   ├── energy/           # Energy pipeline tools
│   ├── storage/          # OPFS shard manager, model loading
│   ├── tooling/          # Shared command contract + runners
│   ├── training/         # Backward/training primitives
│   ├── types/            # Shared type definitions
│   └── utils/            # General utilities
├── benchmarks/           # Vendor benchmark registry + harnesses
├── models/               # Local model artifacts
├── skills/               # Agent skill definitions
├── tools/                # CLI entry points and scripts
├── tests/                # Browser harnesses + kernel tests
├── demo/                 # Browser UI
└── docs/                 # Documentation
```

### Before Starting

- Read `docs/architecture.md` for system overview.
- Read `docs/formats.md` for RDRR format constraints.
- Review `src/inference/pipelines/text.js` for inference flow.
- Review `src/tooling/command-api.js` for command parity contract.

### Command Surfaces (1:1 Contract)

- Browser entry: `runBrowserCommand()` via `src/tooling/browser-command-runner.js`
- Node entry: `runNodeCommand()` via `src/tooling/node-command-runner.js`
- CLI entrypoint: `tools/doppler-cli.js`

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

### Vendor Benchmark Registry

Cross-product benchmark tracking lives under `benchmarks/vendors/`.

- Registry: `benchmarks/vendors/registry.json`
- Workloads: `benchmarks/vendors/workloads.json`
- Capability matrix: `benchmarks/vendors/capabilities.json`
- Harness definitions: `benchmarks/vendors/harnesses/*.json`
- Normalized outputs: `benchmarks/vendors/results/`
- CLI: `tools/vendor-bench.js`

Use these commands when updating benchmark/profiling coverage:

- `node tools/vendor-bench.js validate`
- `node tools/vendor-bench.js capabilities`
- `node tools/vendor-bench.js gap --base doppler --target transformersjs`

When harness/profiling behavior changes (Doppler or vendors), update:
1. harness definition in `benchmarks/vendors/harnesses/`
2. capability matrix in `benchmarks/vendors/capabilities.json`
3. docs in `benchmarks/vendors/README.md`

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

### Core Principles (Policy Contract)

1. Config as code
- Runtime behavior, benchmark methodology, and parity checks must be policy-driven (`*.json`) when practical.
- Avoid ad-hoc runtime switches that only exist in code paths.

2. Explicit over implicit
- Unsupported capability or invalid contract must fail fast with actionable errors.
- Do not silently switch model behavior, kernel path, or benchmark semantics.

3. Contracts first
- Runtime-visible behavior changes must update schema/config/docs in the same change.
- Keep command/API behavior consistent across browser and Node runners.

4. Reproducibility and traceability
- Bench/debug outputs must preserve deterministic knobs (seed, sampling, cache/load mode, kernel path source).
- Apples-to-apples claims require matched workload semantics and explicit mode labeling.

### Execution Plane Contract (Required)

- JSON is the behavior contract. Resolved `manifest.json`, presets, and rule assets must define the runtime decisions before any execution path is entered.
- JavaScript is orchestration: merge/validate config, allocate buffers, copy shards, build pipelines, dispatch work, and read back.
- WGSL is compute only: apply arithmetic using resolved constants/uniforms; no policy branching or command semantics.
- Exceptions are only rule-based and explicit:
  - legacy/compat aliases via registry rule assets
  - capability-gated selection through structured config and rule maps
  - explicit kernel-path overrides in runtime config
- Missing/ambiguous contract must fail fast. No hidden fallbacks for behavior-changing logic.

### Doppler Non-negotiables

1. No silent fallback for command semantics
- Browser and Node surfaces must preserve the same command contract and failure semantics.

2. No hidden benchmark methodology drift
- Benchmark knobs that change comparability must be visible in config/command contracts, not hidden in scripts.

3. Policy files are source-of-truth
- Agent parity policy: `tools/policies/agent-parity-policy.json`
- Vendor benchmark policy: `benchmarks/vendors/benchmark-policy.json`

4. Contract update discipline
- Any runtime-visible metric/field rename must update harness mapping, compare contracts, and docs in the same change.

5. Runtime isolation discipline
- Runtime config/kernel-path changes must be isolated per run and restored on exit.

### Agent Instruction + Skills Parity (Required)

- `AGENTS.md` is the canonical instruction file.
- `CLAUDE.md` must be a symlink to `AGENTS.md`.
- `GEMINI.md` must be a symlink to `AGENTS.md`.
- `skills/` is the canonical skill registry directory.
- `.claude/skills` must be a symlink to `../skills`.
- `.gemini/skills` must be a symlink to `../skills`.
- `.codex/skills` must be a symlink to `../skills`.

Validate parity before committing instruction or skill changes:

- `npm run agents:verify`

### Skills

Canonical path: `skills/` (see `skills/README.md`).

- `doppler-debug`: debug inference issues.
- `doppler-bench`: run performance benchmarks.
- `doppler-perf-squeeze`: investigate and improve decode/prefill performance.
- `doppler-convert`: convert models to RDRR.
- `doppler-kernel-reviewer`: review WGSL/JS kernel implementations against style rules.
- `doppler-bench`, `doppler-convert`, `doppler-debug`, `doppler-kernel-reviewer`, and `doppler-perf-squeeze` are the canonical skill set under `skills/`.

See `docs/config.md` for kernel overrides and runtime modes.
