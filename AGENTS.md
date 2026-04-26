## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Write JavaScript (with `.d.ts`) for the WebGPU inference engine and its unified browser/CLI command surface.

**File Extensions:** Always use `.js`. Never create `.mjs` files. The repo uses `"type": "module"` in `package.json`, so all `.js` files are ES modules. The `.mjs` extension is redundant and prohibited.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent orchestration.

### MANDATORY: Read Style Guides First

Before any non-trivial code edit, read the **invariant files** (compact, must-know rules):

1. **[General Invariants](docs/style/general-style-guide.md#invariants-quick-reference)** — execution plane, no runtime defaults, nullable fields, manifest-first
2. **[JavaScript Invariants](docs/style/javascript-style-guide.md#invariants-quick-reference)** — role boundaries, rule maps, kernel path, failure-path cleanup
3. **[Config Style Guide](docs/style/config-style-guide.md)** — schema layout, merge order, category rules, harness restrictions

For deep work (kernel wrappers, buffer lifecycle, naming conventions, anti-patterns), read the full guides:
- [General Style Guide](docs/style/general-style-guide.md)
- [JavaScript Style Guide](docs/style/javascript-style-guide.md)
- [WGSL Style Guide](docs/style/wgsl-style-guide.md) (shader work only)

### Skill-Specific Style Guide Reads

Baseline for all coding tasks:
- `docs/style/general-style-guide.md` (Invariants quick-reference at top)
- `docs/style/javascript-style-guide.md` (Invariants quick-reference at top)
- `docs/style/config-style-guide.md`

Add full guides by task/skill:
- `doppler-kernel-reviewer`
  - mandatory: `docs/style/general-style-guide.md`, `docs/style/javascript-style-guide.md`, `docs/style/wgsl-style-guide.md`
  - also read `docs/style/config-style-guide.md` when review touches rule selection, dtype policy, or kernel-path metadata
- `doppler-debug`
  - mandatory: `docs/style/general-style-guide.md`, `docs/style/javascript-style-guide.md`, `docs/style/config-style-guide.md`, `docs/style/command-interface-design-guide.md`, `docs/style/harness-style-guide.md`
- `doppler-bench`
  - mandatory: `docs/style/general-style-guide.md`, `docs/style/javascript-style-guide.md`, `docs/style/config-style-guide.md`, `docs/style/command-interface-design-guide.md`, `docs/style/harness-style-guide.md`, `docs/style/benchmark-style-guide.md`
- `doppler-perf`
  - mandatory: `docs/style/general-style-guide.md`, `docs/style/javascript-style-guide.md`, `docs/style/config-style-guide.md`, `docs/style/harness-style-guide.md`, `docs/style/benchmark-style-guide.md`
  - also read `docs/style/wgsl-style-guide.md` for shader changes
  - also read `docs/style/command-interface-design-guide.md` when changing `bench`/`debug` command behavior
- `doppler-convert`
  - mandatory: `docs/style/general-style-guide.md`, `docs/style/javascript-style-guide.md`, `docs/style/config-style-guide.md`, `docs/style/command-interface-design-guide.md`

### Agent Enforcement

- Before the first non-trivial code edit in any turn, open the invariant files above in that turn.
- If work resumes from a summary, handoff, or context compaction, re-open the invariant files before editing.
- When changing config merge, manifest-first resolution, execution-v1, kernel-path selection, or runtime fallback behavior, re-read the full sections in:
  - `docs/style/general-style-guide.md`: `Explicit over Implicit`, `No Runtime Defaults in Code`, `Nullable Required Fields`
  - `docs/style/javascript-style-guide.md`: `Manifest-First Contract`, `Runtime Configuration (Performance Invariants)`
- Any change that could silently rewrite manifest/runtime behavior must either:
  - fail fast with an actionable error, or
  - add a regression test proving the rewrite cannot occur silently.

### Developer Guides (Required For Extension Work)

- `docs/developer-guides/` is the canonical task-oriented playbook layer for extension work.
- These guides are operational checklists, not the normative contract layer.
- Normative rules still live in `docs/style/*.md`, `docs/config.md`, `docs/conversion-runtime-contract.md`, and other core contract docs.

When the task is additive or extension-oriented, open `docs/developer-guides/README.md` and the matching guide before editing. This applies to work such as:
- adding or changing runtime profiles, conversion configs, or kernel-path registries
- adding manifest/runtime fields, chat template formatters, sampling knobs, activations, or kernels
- adding commands, attention variants, quantization formats, or KV-cache layouts
- onboarding a new model family or pipeline family

### Directory Structure

```
doppler/
├── src/
│   ├── cli/              # Public CLI entry point and policy
│   ├── client/           # Client API surface
│   ├── config/           # Schemas and checked-in config registries
│   ├── converter/        # SafeTensors/GGUF → RDRR conversion
│   ├── debug/            # Logging and tracing
│   ├── errors/           # Error types and handling
│   ├── experimental/     # Quarantined experimental/internal-only subsystem lanes
│   ├── formats/          # Format parsing utilities
│   ├── gpu/              # WebGPU device, buffer pools, kernels
│   ├── inference/        # Pipeline, attention, FFN, embeddings
│   ├── generation/       # Text pipeline surface
│   ├── loader/           # GGUF parsing, RDRR manifest
│   ├── memory/           # Heap management, capability detection
│   ├── rules/            # Validation rules (kernel path, etc.)
│   ├── storage/          # OPFS shard manager, model loading
│   ├── tooling/          # Shared command contract + runners
│   ├── types/            # Shared type definitions
│   └── utils/            # General utilities
├── benchmarks/           # Vendor benchmark registry + harnesses
├── models/               # Catalog metadata and external model pointers
├── skills/               # Agent skill definitions
├── tools/                # Internal dev scripts (NOT shipped in package)
├── tests/                # Browser harnesses + kernel tests
├── demo/                 # Browser UI
└── docs/                 # Documentation
    ├── style/            # Style guides + invariant quick-refs
    └── agents/           # Task-specific protocol docs (loaded by skills)
```

Current quarantined subsystem lanes under `src/experimental/` include:
- `adapters/`
- `bridge/`
- `browser/`
- `diffusion/`
- `distribution/`
- `energy/`
- `hotswap/`
- `orchestration/`
- `training/`

### Before Starting

- Read `docs/architecture.md` for system overview.
- Read `docs/rdrr-format.md` for RDRR format constraints.
- Review `src/inference/pipelines/text.js` for inference flow.
- Review `src/tooling/command-api.js` for command parity contract.
- For extension work, read `docs/developer-guides/README.md` and the matching guide in `docs/developer-guides/`.

### Public vs Internal Tooling

See `docs/agents/tooling-surface.md` for the full breakdown.

- **Public CLI**: `src/cli/doppler-cli.js` — ships as `bin.doppler`
- **Command infrastructure**: `src/tooling/` — partially exported via `./tooling`
- **Dev scripts**: `tools/` — internal repo scripts, never shipped (except `tools/convert-safetensors-node.js`)

### Command Surfaces (1:1 Contract)

- Browser entry: `runBrowserCommand()` via `src/tooling/browser-command-runner.js`
- Node entry: `runNodeCommand()` via `src/tooling/node-command-runner.js`
- CLI entrypoint: `src/cli/doppler-cli.js`

Rules:
- New commands must be added to `src/tooling/command-api.js`.
- Command semantics must match on browser and CLI.
- Unsupported environment capabilities must fail fast (never silent fallback).

### CLI Quick Reference

Commands have workload/intent rules defined in `src/rules/tooling/command-runtime.rules.json`. Do not guess the workload — use the table below.

| Command   | Workload          | Intent        | Example |
|-----------|-------------------|---------------|---------|
| `bench`   | caller choice     | `calibrate`   | `node src/cli/doppler-cli.js bench --config '{"request":{"workload":"inference","modelId":"gemma3-270m"}}' --json` |
| `debug`   | caller choice     | `investigate` | `node src/cli/doppler-cli.js debug --config '{"request":{"workload":"inference","modelId":"gemma3-270m"}}' --json` |
| `verify`  | caller choice     | `verify`      | `node src/cli/doppler-cli.js verify --config '{"request":{"workload":"inference","modelId":"gemma3-270m"}}' --json` |
| `convert` | n/a               | —             | `node src/cli/doppler-cli.js convert --config <path|url|json>` |
| `lora`    | n/a               | —             | `node src/cli/doppler-cli.js lora --config <path|url|json>` |
| `distill` | n/a               | —             | `node src/cli/doppler-cli.js distill --config <path|url|json>` |

- `--config` accepts inline JSON, file path, or URL for all commands.
- The CLI auto-resolves models from the external RDRR root (`/Volumes/models/rdrr` on macOS, `/media/x/models/rdrr` on Linux) by `modelId`.
- To point at a model outside the external root, set `request.modelUrl` to a `file://` path. `modelUrl` is a **request-level** field.
- Use `--surface node` to force Node/WebGPU, `--surface browser` to force headless Chromium, or omit for `auto`.

### Config System

Use runtime profiles/config payloads, not ad-hoc per-field flags.

- Runtime profiles: `src/config/runtime/profiles/`
- Conversion configs: `src/config/conversion/` (v1 format with inline execution graph)
- Read tunables via `getRuntimeConfig()`; avoid hardcoded defaults in runtime paths.
- `runtime.shared.tooling.intent` is required for harnessed debug/bench/test flows.

### Task-Specific Protocols

These docs are loaded by skills on demand — not for every task:

- `docs/agents/conversion-protocol.md` — conversion triage + promotion gate (used by `doppler-convert`, `doppler-debug`)
- `docs/agents/debug-protocol.md` — inference debug ladder (used by `doppler-debug`)
- `docs/agents/benchmark-protocol.md` — vendor benchmark registry (used by `doppler-bench`, `doppler-perf`)
- `docs/agents/hardware-notes.md` — GPU memory assumptions (used by all inference skills)

See `docs/agents/README.md` for the full index.

### Performance Investigation Discipline

- Treat prefill and decode as separate phases. Dump resolved kernel path, active decode mode, and loaded weight/materialization dtype per phase before changing kernels.
- Keep parity and throughput lanes separate. Claimable compare work belongs to fairness-managed parity lanes; throughput-tuned runs are tuning evidence until promoted explicitly.
- Before kernel edits, classify the wall: GPU compute vs submit/readback/orchestration. Do not assume a math-kernel bug when timing already shows orchestration dominating.

### Correctness Regression Discipline

When a code change (kernel swap, transform, dtype policy, materialization path) produces incorrect model output — garbled text, numerical divergence, repeated tokens, or silent quality loss:

1. **Hypothesis first.** Before reading code, write down exactly one falsifiable hypothesis and the diagnostic that would disprove it. Do not read more than 5 files before running a diagnostic.
2. **Measure before tracing.** After isolating a behavioral difference (e.g., "path A correct, path B wrong"), the next step is always a per-layer or per-op numerical readback — not more static code analysis. Use `doppler-debug` probes to capture boundary values at the first divergence point.
3. **Scope the change, not the codebase.** If the regression correlates with a specific transform or kernel swap, diff the execution graph before and after the transform. Do not trace the entire pipeline hoping to find the bug by inspection.
4. **Do not ship workarounds as fixes.** Disabling a broken path is a workaround, not a fix. Label it as such. The underlying bug must be tracked and the workaround must include a comment or test explaining what it avoids and why.
5. **Auto-trigger `doppler-debug`.** When the task involves incorrect model output, garbled text, or numerical divergence, invoke `doppler-debug` before any manual investigation. The debug ladder (classify → reference → dump → compare → isolate) exists to prevent aimless code reading.

### Logging

Use debug module (`src/debug/index.js`), not raw `console.*` in runtime code.

Allowed direct console output: `tools/` entry points, `tests/` harnesses, `demo/` entry points, one-time startup in `src/gpu/device.js`.

Do not add throwaway log statements. Use existing trace categories, config-driven probes, or permanent trace extensions. See [General Invariants](docs/style/general-style-guide.md#invariants-quick-reference).

### Guardrails

- Handle GPU device loss gracefully.
- Validate tensor shapes at kernel boundaries.
- Keep command/runtime behavior deterministic under fixed config.
- Do not ship surface-specific command behavior drift.

### Core Principles (Policy Contract)

1. Config as code
- Runtime behavior, benchmark methodology, and parity checks must be policy-driven (`*.json`) when practical.

2. Explicit over implicit
- Unsupported capability or invalid contract must fail fast with actionable errors.
- Do not silently switch model behavior, kernel path, or benchmark semantics.

3. Contracts first
- Runtime-visible behavior changes must update schema/config/docs in the same change.
- Keep command/API behavior consistent across browser and Node runners.

4. Reproducibility and traceability
- Bench/debug outputs must preserve deterministic knobs (seed, sampling, cache/load mode, kernel path source).

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
- `doppler-perf`: investigate and improve decode/prefill performance.
- `doppler-convert`: convert models to RDRR.
- `doppler-kernel-reviewer`: review WGSL/JS kernel implementations against style rules.

See `docs/config.md` for kernel overrides and runtime modes.

## No time estimates

- never estimate work in hours, days, weeks, or any other time unit, in code, comments, commit messages, status updates, receipts, or chat replies
- do not say "~30 min", "~2 hr", "multi-day", "quick", "long-running" as size proxies for engineering work
- describe what the work IS — the file to change, the function to add, the schema field to extend, the named blocker to fix — not how long it should take
- if scope must be conveyed, list the concrete deltas (lines/files/symbols touched) instead of a duration
