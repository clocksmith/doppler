## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Build Doppler as a dependable JavaScript library for local
WebGPU inference: easy to embed, fast to start and execute, and safe to reuse.
Rig is the supporting model compiler/preparation system producing source-truth,
qualified, signed immutable Capsules; the deliberately uncreative Run validates,
selects, binds, executes, and observes their declared programs. Browser and Node
support are qualified separately; Bun remains experimental.

**Current priorities:** Work within the product boundaries in
[GOALS.md](GOALS.md#current-priorities). The maintainers' ordered roadmap is
private (in the deco workspace: `../ouroboros/docs/projects/doppler/roadmap.md`);
public contributors follow GitHub issues and GOALS.md. Reploid is a demanding
consumer, not the owner of Doppler's roadmap; Doe is optional. Historical search,
rewriting, and provider campaigns are retained evidence and acceptance contracts,
not mandatory next work. Preserve model identity, compatibility, trust/checkpoint
protections, and scoped qualification. Verification supports useful, fast,
simple, portable execution.

**File Extensions:** Always use `.js`; never create `.mjs`. `package.json` sets `"type": "module"`, so every `.js` file is already an ES module.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent orchestration.

## Git workflow: direct to main

- Commit on the owning repo's `main` and push to `origin/main`. No feature
  branches, branch-backed worktrees, or PRs unless the user asks for them.
- On another branch, preserve its work and move the task to `main` safely;
  never discard changes to switch. Never force-push `main`.
- Stage only task-related changes; integrate remote updates without
  overwriting unrelated work.
- Report the pushed commit or the concrete blocker. A local commit is not a push.

### MANDATORY: Read Style Guides First

Before a non-trivial edit, read the [engineering checklist](docs/style/general-style-guide.md#invariants-quick-reference).
Then read the linked sections and JavaScript/config invariants relevant to the
changed behavior. Full guides remain references, not a mandatory 2,200-line
preflight for an ordinary edit. Deep kernel, buffer-lifecycle, config-resolution
or shader work still reads the applicable full guides below.

### Skill-Specific Style Guide Reads

Baseline for coding tasks: the engineering checklist and affected invariants. Add full guides
by skill (`general`, `javascript`, `config`, `wgsl`, `harness`, `benchmark` =
`docs/style/<name>-style-guide.md`; `command-interface` =
`docs/style/command-interface-design-guide.md`):
- `doppler-kernel-reviewer`: general, javascript, wgsl; plus config when review
  touches rule selection, dtype policy, or kernel-path metadata
- `doppler-debug`: general, javascript, config, command-interface, harness
- `doppler-bench`: general, javascript, config, command-interface, harness, benchmark
- `doppler-perf`: general, javascript, config, harness, benchmark; plus wgsl for
  shader changes and command-interface when changing `bench`/`debug` behavior
- `doppler-convert`: general, javascript, config, command-interface

### Agent Enforcement

- Open the engineering checklist and affected invariant sections before the first non-trivial code edit in each turn,
  and again when resuming from a summary, handoff, or context compaction.
- When changing config merge, manifest-first resolution, execution-v1, kernel-path selection, or runtime fallback behavior, re-read the full sections in:
  - `docs/style/general-style-guide.md`: `Explicit over Implicit`, `No Runtime Defaults in Code`, `Nullable Required Fields`
  - `docs/style/javascript-style-guide.md`: `Manifest-First Contract`, `Runtime Configuration (Performance Invariants)`
- A change that could silently rewrite manifest/runtime behavior must fail fast
  with an actionable error or add a regression test proving it cannot.

### Developer Guides (Required For Extension Work)

`docs/developer-guides/` holds task-oriented playbooks (operational checklists,
not normative contracts; those stay in `docs/style/*.md`, `docs/config.md`,
`docs/conversion-runtime-contract.md`, and other core contract docs). Before
additive or extension work, open `docs/developer-guides/README.md` and the
matching guide: adding or changing runtime profiles, conversion configs,
kernel-path registries, manifest/runtime fields, chat template formatters,
sampling knobs, activations, kernels, commands, attention variants,
quantization formats, or KV-cache layouts, or onboarding a model or pipeline family.

For model adoption, follow the [depth and reusable support checklist](docs/developer-guides/model-onboarding-playbook.md#depth-and-reusable-support).
Keep component support, artifact preparation and full-model qualification
distinct in status reports. Reuse semantic owners and declared capability
contracts; a profile copied from a smaller model is a candidate, not evidence
that its kernels admit the new model's geometry or precisions.

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
│   ├── experimental/     # Quarantined internal-only lanes: adapters, bridge, browser, diffusion, distribution, energy, hotswap, orchestration, training
│   ├── formats/          # Format parsing utilities
│   ├── gpu/              # WebGPU device, buffer pools, kernels
│   ├── inference/        # Pipeline, attention, FFN, embeddings
│   ├── generation/       # Text pipeline surface
│   ├── loader/           # GGUF parsing, RDRR manifest
│   ├── memory/           # Heap management, capability detection
│   ├── rules/            # Validation rules (kernel path, etc.)
│   ├── storage/          # OPFS shard manager, model loading
│   ├── tooling/          # Shared command contract + runners
│   └── types/            # Shared type definitions
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

### Before Starting

- Read `docs/architecture.md` (system overview) and `docs/rdrr-format.md` (RDRR format constraints).
- Review the text inference flow and `src/tooling/command-api.js` (command parity contract). `src/inference/pipelines/text.js` only registers pipeline factories by `modelType`; the flow lives under `src/inference/pipelines/text/`: `pipeline.js` (class), `init.js` and `lifecycle.js` (load), `generator.js` (prefill/decode loop) and `layer.js` (per-layer execution).

### Component Intent

Before modifying a file, read the canonical project goals and every
`CATSCAN.md` from the project root to the target directory, parent first (the
union of chains for changes spanning directories).

Target, Authority, Scope, Contracts, Invariants, Acceptance, and Non-goals are
implementation constraints. A child CATSCAN may narrow its parent but never
silently broaden authority, weaken an invariant, or contradict it. Existing code
does not overrule a charter; code may itself have drifted.

Explicit user direction may change component intent: identify the boundary
change and update the affected CATSCAN with the implementation. Never rewrite a
charter to excuse failing code or evidence. CATSCAN constrains outcomes and
authority, not internal algorithms.

Repository-change handoffs must state:

- `Component: <component ID>`
- `Intent: preserved | changed`
- `Acceptance evidence: <commands or artifacts>`
- `Boundary effects: none | <named components>`

When you change a charter, component boundary, or the CATSCAN policy inventory,
run `npm run catscan:check`.

### Public vs Internal Tooling

Full breakdown: `docs/agents/tooling-surface.md`.

- **Public CLI**: `src/cli/doppler-cli.js` — ships as `bin.doppler`
- **Command infrastructure**: `src/tooling/` — partially exported via `./tooling`
- **Dev scripts**: `tools/` — internal, never shipped (except `tools/convert-safetensors-node.js`)

### Command Surfaces (1:1 Contract)

- Browser entry: `runBrowserCommand()` via `src/tooling/browser-command-runner.js`
- Node entry: `runNodeCommand()` via `src/tooling/node-command-runner.js`
- CLI entrypoint: `src/cli/doppler-cli.js`

New commands must be added to `src/tooling/command-api.js`. Command semantics
must match on browser and CLI; unsupported environment capabilities fail fast
(never silent fallback).

### CLI Quick Reference

Workload/intent rules live in `src/rules/tooling/command-runtime.rules.json`. Do not guess the workload; use this table.

| Command | Workload | Intent | Example |
|---|---|---|---|
| `bench` | caller choice | `calibrate` | `node src/cli/doppler-cli.js bench --config '{"request":{"workload":"inference","modelId":"gemma3-270m"}}' --json` |
| `debug` | caller choice | `investigate` | `node src/cli/doppler-cli.js debug --config '{"request":{"workload":"inference","modelId":"gemma3-270m"}}' --json` |
| `diagnose` | caller choice | `investigate` | `node src/cli/doppler-cli.js diagnose --config '{"request":{"workload":"inference","modelId":"gemma3-270m"}}' --json` |
| `verify` | caller choice | `verify` | `node src/cli/doppler-cli.js verify --config '{"request":{"workload":"inference","modelId":"gemma3-270m"}}' --json` |
| `convert` | n/a | — | `node src/cli/doppler-cli.js convert --config <path|url|json>` |
| `profiles` | n/a | discovery | `node src/cli/doppler-cli.js profiles --json` |
| `lora` | n/a | — | `node src/cli/doppler-cli.js lora --config <path|url|json>` |
| `distill` | n/a | — | `node src/cli/doppler-cli.js distill --config <path|url|json>` |

- `--config` accepts inline JSON, file path, or URL for all commands.
- The CLI auto-resolves models from the external RDRR root (`/Volumes/models/rdrr` on macOS, `/media/x/models/rdrr` on Linux) by `modelId`.
- For a model outside that root, set the **request-level** `request.modelUrl` to a `file://` path.
- `--surface node` forces Node/WebGPU, `--surface browser` forces headless Chromium; omit for `auto`.
- Use `doppler profiles --json` to discover checked-in runtime profile IDs before choosing an investigation, verify, or calibration profile.
- `--runtime-profile <id>` is CLI shorthand for `request.runtimeProfile` (full JSON config stays canonical); never combine it with `--runtime-config` or runtime fields inside `--config`.

### Config System

Use runtime profiles/config payloads, not ad-hoc per-field flags.

- Runtime profiles: `src/config/runtime/profiles/`
- Conversion configs: `src/config/conversion/` (v1 format with inline execution graph)
- Read tunables via `getRuntimeConfig()`; avoid hardcoded defaults in runtime paths.
- `request.intent`, normalized from command metadata, is the sole active-run
  intent authority. Runtime profiles declare top-level `compatibleIntents`; never
  author the removed `runtime.shared.tooling.intent`.
- Experimental model-scoped dtype lanes must be expressed as a runtime profile
  plus an execution-v1 capability transform. Do not change catalog/release
  metadata for the lane until the matching browser/WebGPU transcript or parity
  receipt exists.

### Capability Transform / Dtype Policy

Before changing dtype fallback, execution-v1 capability transforms, kernel-path
remap behavior, or model-scoped runtime lanes, classify the rule in
`src/rules/inference/capability-transforms.rules.json`:

- `hardware-compatibility` (missing device feature only) and
  `runtime-session-compatibility` (resolved session constraint such as
  `kvDtype=f32`): no model/platform identity.
- `explicit-lane`: requested f16/selective-f16 lane from a runtime profile or manifest variant; not a fallback.
- `platform-workaround`: exact platform plus exact model/list and evidence; no silent dtype change.
- `lane-mismatch-guard`: fail closed when profile and manifest variant disagree.
- `capability-optimization`: positive capability optimization, no dtype change.
- `default-noop`: final empty-match rule.

Every rule must include `kind`, `dtypeEffect`, `reason`, and `evidence`;
execution receipts must expose the matched policy. When you edit these rules,
run `npm run capability-policy:check`. If behavior cannot be classified, fail
fast instead of adding a silent fallback.

### Task-Specific Protocols

Loaded by skills on demand, not for every task (full index: `docs/agents/README.md`):

- `docs/agents/conversion-protocol.md` — conversion triage + promotion gate (`doppler-convert`, `doppler-debug`)
- `docs/agents/debug-protocol.md` — inference debug ladder (`doppler-debug`)
- `docs/agents/benchmark-protocol.md` — vendor benchmark registry (`doppler-bench`, `doppler-perf`)
- `docs/agents/hardware-notes.md` — GPU memory assumptions (all inference skills)

### Performance Investigation Discipline

- Treat prefill and decode as separate phases; dump each phase's resolved kernel path, active decode mode, and loaded weight/materialization dtype before changing kernels.
- Keep parity and throughput lanes separate. Claimable compare work belongs to fairness-managed parity lanes; throughput-tuned runs are tuning evidence until promoted explicitly.
- Before kernel edits, classify the wall: GPU compute vs submit/readback/orchestration. Do not assume a math-kernel bug when timing shows orchestration dominating.

### Intent-First Operations

- Treat Doppler intent as real model/runtime capability and fair benchmark evidence, not README wording.
- If the user asks what works, produce a scoreboard from current registry entries, manifests, sidecars, reports, and claim indexes before summarizing.
- Preserve exact model/platform scope: browser-only does not imply Node; named embedding/reranker models come before adjacent generation models.
- When the user distrusts a Doppler-vs-TJS or vendor benchmark, inspect and fix the harness fairness gates before editing public claims.
- Fairness output must expose comparable surface, blocked lanes, overridden settings, failed runs, throughput-only rows, model identity, backend identity, and artifact paths.
- Do not publish win/loss language until the claim-grade gate has mechanically passed.
- Answer deployment/package status from `package.json`, npm/local package artifacts, source version files, and actual publish/deploy command output.

### Correctness Regression Discipline

When a code change (kernel swap, transform, dtype policy, materialization path) produces incorrect model output (garbled text, numerical divergence, repeated tokens, silent quality loss):

1. **Auto-trigger `doppler-debug`** before any manual investigation. Its ladder (classify → reference → dump → compare → isolate) captures the first numerical or token divergence before broad runtime refactoring.
2. **Hypothesis first.** Write down one falsifiable hypothesis and the diagnostic that would disprove it. Read at most 5 files before running a diagnostic.
3. **Measure before tracing.** Once a behavioral difference is isolated ("path A correct, path B wrong"), take a per-layer or per-op numerical readback with `doppler-debug` probes at the first divergence, not more static analysis.
4. **Scope the change, not the codebase.** If the regression correlates with a transform or kernel swap, diff the execution graph before and after it instead of tracing the whole pipeline.
5. **Workarounds are not fixes.** Label a disabled broken path as a workaround, track the underlying bug, and add a comment or test stating what it avoids and why.

### Logging

Use the debug module (`src/debug/index.js`), not raw `console.*`, in runtime code. Direct console output is allowed only in `tools/` entry points, `tests/` harnesses, `demo/` entry points, and one-time startup in `src/gpu/device.js`.

No throwaway log statements: use existing trace categories, config-driven probes, or permanent trace extensions.

### Guardrails

- Handle GPU device loss gracefully.
- Validate tensor shapes at kernel boundaries.
- Keep command/runtime behavior deterministic under fixed config.

### Core Principles (Policy Contract)

1. Config as code
- Runtime behavior, benchmark methodology, and parity checks must be policy-driven (`*.json`) when practical.

2. Explicit over implicit
- Unsupported capability or invalid contract must fail fast with actionable errors.
- Do not silently switch model behavior, kernel path, or benchmark semantics.

3. Contracts first
- Runtime-visible behavior changes must update schema/config/docs in the same change.

4. Reproducibility and traceability
- Bench/debug outputs must preserve deterministic knobs (seed, sampling, cache/load mode, kernel path source).

### Doppler Non-negotiables

1. No silent fallback for command semantics
- Browser and Node surfaces must preserve the same command contract and failure semantics; no surface-specific drift.

2. No hidden benchmark methodology drift
- Benchmark knobs that change comparability must be visible in config/command contracts, not hidden in scripts.

3. Policy files are source-of-truth
- Agent parity policy: `tools/policies/agent-parity-policy.json`
- Vendor benchmark policy: `benchmarks/vendors/benchmark-policy.json`

4. Contract update discipline
- Any runtime-visible metric/field rename must update harness mapping, compare contracts, and docs in the same change.

5. Run isolation discipline
- Runtime config/kernel-path changes must be isolated per run and restored on exit.

### Agent Instruction + Skills Parity (Required)

- `AGENTS.md` is the canonical instruction file; `CLAUDE.md` and `GEMINI.md`
  must be symlinks to it. Edit `AGENTS.md`, never the symlinks.
- `skills/` is the canonical skill registry; `.agents/skills`, `.claude/skills`,
  `.gemini/skills`, and `.codex/skills` must be symlinks to `../skills`.

Run `npm run agents:verify` only when adding, moving, or recreating these
symlinks or a `skills/` entry, or when renaming or removing an AGENTS.md heading
listed in `tools/policies/agent-parity-policy.json`. Text edits need no check.

### Skills

Index: `skills/README.md`.

- `doppler-debug`: debug inference issues.
- `doppler-bench`: run performance benchmarks.
- `doppler-perf`: investigate and improve decode/prefill performance.
- `doppler-convert`: convert models to RDRR.
- `doppler-kernel-reviewer`: review WGSL/JS kernel implementations against style rules.

See `docs/config.md` for kernel overrides and runtime modes.

## Development verification

- During development, choose checks for the changed behavior: syntax/type checks,
  existing focused tests, or a local visual check. A command list is not a requirement
  to run every suite before each commit, push, or ordinary application deployment.
- Do not automatically run full repository suites, broad browser/device matrices,
  GPU/model campaigns, benchmarks, or full release qualification. Run them when
  explicitly requested or when a named correctness question cannot be answered by
  focused checks; state that reason before starting.
- Config-contract, mirror, drift, and freshness checks run only when the change
  edits what they check. Don't add tests that just restate another file or pin
  config literals. Time-box long processes.
- Once relevant checks pass, continue. Repeat or broaden verification only after
  another relevant change, a failure, or a concrete unresolved concern.
- For documentation-only edits, inspect the diff and changed links. Do not run
  application, GPU, browser, or benchmark suites for prose changes.
- For ordinary deployments, build the changed artifact where needed and check the
  affected route after deployment. Deployment alone does not require a new release
  qualification campaign.
- Keep existing acceptance thresholds and release-qualification requirements.
  Report exactly what was checked; focused development checks do not establish
  numerical qualification, performance, multi-device support, or full acceptance.

## No speculative engineering timelines

- Never predict how long engineering work will take ("1-2 weeks", "a quick
  fix"). Describe concrete deltas, dependencies, risks, and validation instead.
- For an already-running process, report measured elapsed time, progress, or a
  grounded ETA only when the process exposes one; otherwise report its phase,
  latest output, and whether it is still progressing.

## Pick the real fix

- Fix correctness bugs; don't relabel them. Effort or scope framing
  ("non-trivial", "worth its own thread", "later") is no reason for a lesser fix.
- "Mark experimental", "add a TODO", or "reword the misleading comment" does not
  substitute for fixing wrong behavior.
- If scope must be split, list the concrete deltas and ask the user which path
  to take; don't pre-decide a smaller version.
