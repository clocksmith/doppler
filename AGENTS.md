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

Read only the owner documents for the change at hand; there is no repository-wide reading list.

| Change | Read at the owner |
| --- | --- |
| Any non-trivial code edit | [Engineering checklist](docs/style/general-style-guide.md#invariants-quick-reference), then the invariant sections it links for the changed behavior |
| Config merge, manifest-first resolution, execution-v1, kernel-path selection, runtime fallback | General guide: `Explicit over Implicit`, `No Runtime Defaults in Code`, `Nullable Required Fields`; JavaScript guide: `Manifest-First Contract`, `Runtime Configuration (Performance Invariants)`; [config guide](docs/style/config-style-guide.md) |
| Kernels or shaders | [WGSL guide](docs/style/wgsl-style-guide.md), [JavaScript guide](docs/style/javascript-style-guide.md); `doppler-kernel-reviewer` adds the config guide when review touches rule selection, dtype policy or kernel-path metadata |
| Debugging inference output | [Debug protocol](docs/agents/debug-protocol.md), [hardware notes](docs/agents/hardware-notes.md), [harness guide](docs/style/harness-style-guide.md), [command-interface guide](docs/style/command-interface-design-guide.md) |
| Benchmarks or performance | [Benchmark protocol](docs/agents/benchmark-protocol.md), [benchmark guide](docs/style/benchmark-style-guide.md), [harness guide](docs/style/harness-style-guide.md); the WGSL guide for shader changes and the command-interface guide when `bench`/`debug` behavior changes |
| Conversion | [Conversion protocol](docs/agents/conversion-protocol.md), [config guide](docs/style/config-style-guide.md), [command-interface guide](docs/style/command-interface-design-guide.md) |
| Adding or changing runtime profiles, conversion configs, kernel-path registries, manifest/runtime fields, chat template formatters, sampling knobs, activations, kernels, commands, attention variants, quantization formats or KV-cache layouts; onboarding a model or pipeline family | [Developer guides index](docs/developer-guides/README.md) and the matching playbook; for model adoption, the [depth and reusable support checklist](docs/developer-guides/model-onboarding-playbook.md#depth-and-reusable-support) |
| Text inference flow | `docs/architecture.md`, `docs/rdrr-format.md`, `src/tooling/command-api.js` (command parity contract); `src/inference/pipelines/text.js` only registers factories by `modelType`, the flow lives under `src/inference/pipelines/text/` in `pipeline.js`, `init.js`, `lifecycle.js`, `generator.js` and `layer.js` |

Developer guides are operational checklists, not normative contracts; contracts
stay in `docs/style/*.md`, `docs/config.md`, `docs/conversion-runtime-contract.md`
and the other core contract docs. Protocol documents under `docs/agents/` load on
demand through skills (index: `docs/agents/README.md`), not for every task.

Re-open the checklist and the affected invariant sections before the first
non-trivial code edit in each turn, and again when resuming from a summary,
handoff or context compaction. A change that could silently rewrite
manifest/runtime behavior must fail fast with an actionable error or add a
regression test proving it cannot. Keep component support, artifact preparation
and full-model qualification distinct in status reports; a profile copied from a
smaller model is a candidate, not evidence that its kernels admit the new model's
geometry or precisions.

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
│   ├── tooling-exports/  # Narrow public tooling facades
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

The diagram is an overview. The complete owner and root-facade inventory is
`tools/policies/source-architecture-policy.json` (`topLevelOwners`, `rootFileOwners`);
for example, `src/capsule-runtime.js` and `src/training.js` are root facades.

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

Workload/intent rules live in `src/rules/tooling/command-runtime.rules.json`; do
not guess the workload. The command table, `--config` forms, model resolution,
`--surface` and `--runtime-profile` are documented in [docs/cli.md](docs/cli.md).

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
`src/rules/inference/capability-transforms.rules.json` using the
[capability transform categories](docs/style/config-style-guide.md#capability-transform-categories).
Every rule must include `kind`, `dtypeEffect`, `reason`, and `evidence`;
execution receipts must expose the matched policy. When you edit these rules,
run `npm run capability-policy:check`. If behavior cannot be classified, fail
fast instead of adding a silent fallback.

### Performance Investigation Discipline

Follow the [benchmark protocol](docs/agents/benchmark-protocol.md): prefill and
decode are separate phases, parity and throughput lanes stay separate, and the
wall is classified (GPU compute versus submit/readback/orchestration) before any
kernel edit.

### Intent-First Operations

- Treat Doppler intent as real model/runtime capability and fair benchmark evidence, not README wording.
- If the user asks what works, produce a scoreboard from current registry entries, manifests, sidecars, reports, and claim indexes before summarizing.
- Preserve exact model/platform scope: browser-only does not imply Node; named embedding/reranker models come before adjacent generation models.
- When the user distrusts a Doppler-vs-TJS or vendor benchmark, inspect and fix the harness fairness gates before editing public claims.
- Fairness output must expose comparable surface, blocked lanes, overridden settings, failed runs, throughput-only rows, model identity, backend identity, and artifact paths.
- Do not publish win/loss language until the claim-grade gate has mechanically passed.
- Answer deployment/package status from `package.json`, npm/local package artifacts, source version files, and actual publish/deploy command output.

### Correctness Regression Discipline

When a code change (kernel swap, transform, dtype policy, materialization path)
produces incorrect model output (garbled text, numerical divergence, repeated
tokens, silent quality loss), auto-trigger `doppler-debug` before any manual
investigation and follow its
[regression steps](docs/agents/debug-protocol.md#8-regression-after-a-code-change).
Workarounds are not fixes: label a disabled broken path as a workaround, track
the underlying bug, and add a comment or test stating what it avoids and why.

### Logging

Use the debug module (`src/debug/index.js`), not raw `console.*`, in runtime code. Direct console output is allowed in `src/cli/` for the requested command result or diagnostic output, and in `tools/` entry points, `tests/` harnesses, `demo/` entry points, and one-time startup in `src/gpu/device.js`.

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
