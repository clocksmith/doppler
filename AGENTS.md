## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

When this repository is checked out under Deco, `../AGENTS.md` supplies the
shared operating rules. This file covers Doppler-specific commands and contracts;
a standalone checkout does not require the Deco workspace.

**Prime Directive:** Build Doppler as a dependable JavaScript library for local
WebGPU inference; Rig prepares signed Capsules and Run executes them unchanged.

**Current priorities:** [GOALS.md](GOALS.md#current-priorities) owns mission and
priorities. The maintainers' ordered roadmap is private (in the deco workspace:
`../ouroboros/docs/projects/doppler/roadmap.md`); public contributors follow
GitHub issues and GOALS.md. Reploid is a consumer, not the owner of Doppler's
roadmap. Retained campaigns are evidence, not mandatory next work.

**File Extensions:** Always use `.js`; never create `.mjs`. `package.json` sets `"type": "module"`, so every `.js` file is already an ES module. Code is JavaScript only (WGSL shaders are fine): no Python or Go files, and no scripts that run them. `npm run js-only:check`, part of `check:fast`, enforces this; run vendor or reference stacks that need Python outside the repository and import their JSON.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent orchestration.

### Style guides

**Commit and push to `main`.** Every verified change goes straight to `main`: rebase, push,
confirm CI is green, fix forward if it goes red. Do not open pull requests; if a direct push
is refused, open one and merge it immediately yourself, then delete the branch. Never leave a
PR or branch waiting for review. (Publishing to npm and release tags still need the owner.)

Match the check to the risk. Times are from one laptop; CI runs on Linux.

| When | Run | Time |
| --- | --- | --- |
| Editing, once the behavior is ready to verify | The test file for the loader, pipeline or kernel you changed: `node tools/run-node-tests.js <file>` | under 2 s a file |
| Before every push | `npm run check:fast` (source style, architecture, changed-file style rules, artifact retention, kebab-case enums, JavaScript-only); `npm run typecheck:source` when declarations or exports changed | 22 s; 7 s |
| Before pushing to `main` | The contract tests for the behavior you changed (below) and `npm run check:changed`, which runs only the policy checks your files select | seconds |
| Every push to `main` | CI `Smoke` (`.github/workflows/check-green.yml`): contract tests, declaration types, `check:fast`, style guides, the installed-package smoke. No GPU, browser or model download | 2.5 min |
| Release, hardware work, or a change that justifies it | `npm run check:green` and `npm run test:unit` (the whole unit suite), `test:unit:webgpu`, `test:gpu`, the browser harness and the manual workflows | not in CI |

Contract tests that stay when their behavior changes:
command and error contracts (`tests/inference/execution-backend-contract.test.js`,
`tests/config/error-taxonomy.test.js`, `tests/config/public-error-contract.test.js`),
config ownership (`tests/config/runtime-transform-ownership.test.js`,
`tests/integration/execution-graph-transforms.test.js`), Capsule trust
(`tests/capsule/`, `tests/runtime/capsule-artifact-source.test.js`), cancellation
(`tests/inference/pipeline-load-cancellation.test.js`,
`tests/inference/pipeline-scope-cancellation.test.js`,
`tests/inference/generator-abort-delivery.test.js`), the consumer boundary Reploid calls
(`tests/integration/doppler-scoped-session.test.js`,
`tests/integration/doppler-browser-api-parity.test.js`, `npm run package:smoke`) and the
privacy protocol (`npm test --prefix tools/privacy-demo`). Model parity is qualified on
hardware, not by these.

**What opens a GPU device.** A Node test gets a device only through the optional `webgpu`
or `doe-gpu` package, so a checkout installed with `npm ci --omit=optional`, as CI installs,
cannot open one (device tests fail there instead). In a default install, check a file before
running it on a machine whose GPU someone is timing:
`grep -lE "bootstrapNodeWebGPU|requestAdapter\(|initDevice\(|getDevice\(" <files>`. The
`unit-webgpu` suite in `tools/policies/test-coverage-policy.json` lists 37 such files, but
about 70 more still sit in the `unit` suite, so `npm test`, `check:behavior` and
`check:green` open a device. `node tools/run-node-tests.js --suite unit --list` prints
the selection without running it.

The [guide lookup](docs/style/README.md#guide-lookup),
[rule coverage](tools/policies/style-guide-rules.json) and
[engineering checklist](docs/style/general-style-guide.md#invariants-quick-reference)
are references: open the sections that bear on the change. Deep kernel,
buffer-lifecycle, config-resolution or shader work benefits from the applicable
full guides below.

Owner documents are lookups for the change at hand; there is no reading list.

| Change | Relevant owner reference |
| --- | --- |
| Cross-cutting runtime behavior | [Engineering checklist](docs/style/general-style-guide.md#invariants-quick-reference) and the relevant invariant sections |
| Config merge, manifest-first resolution, execution-v1, kernel-path selection, runtime fallback | General guide: `Explicit over Implicit`, `No Runtime Defaults in Code`, `Nullable Required Fields`; JavaScript guide: `Manifest-First Contract`, `Runtime Configuration (Performance Invariants)`; [config guide](docs/style/config-style-guide.md) |
| Kernels or shaders | [WGSL guide](docs/style/wgsl-style-guide.md), [JavaScript guide](docs/style/javascript-style-guide.md); `doppler-kernel-reviewer` adds the config guide when review touches rule selection, dtype policy or kernel-path metadata |
| Debugging inference output | [Debug protocol](docs/agents/debug-protocol.md), [hardware notes](docs/agents/hardware-notes.md), [harness guide](docs/style/harness-style-guide.md), [command-interface guide](docs/style/command-interface-design-guide.md) |
| Benchmarks or performance | [Benchmark protocol](docs/agents/benchmark-protocol.md), [benchmark guide](docs/style/benchmark-style-guide.md), [harness guide](docs/style/harness-style-guide.md); the WGSL guide for shader changes and the command-interface guide when `bench`/`debug` behavior changes |
| Conversion | [Conversion protocol](docs/agents/conversion-protocol.md), [config guide](docs/style/config-style-guide.md), [command-interface guide](docs/style/command-interface-design-guide.md) |
| Adding or changing runtime profiles, conversion configs, kernel-path registries, manifest/runtime fields, chat template formatters, sampling knobs, activations, kernels, commands, attention variants, quantization formats or KV-cache layouts; onboarding a model or pipeline family | [Developer guides index](docs/developer-guides/README.md) and the matching playbook; for model adoption, the [depth and reusable support checklist](docs/developer-guides/model-onboarding-playbook.md#depth-and-reusable-support) |
| Text inference flow | `docs/architecture.md`, `docs/rdrr-format.md`, `src/tooling/command-api.js` (command parity contract); `src/inference/pipelines/text.js` registers factories by `modelType`. Under `text/`, `pipeline.js` composes the API, `init.js` forwards initialization, `lifecycle.js` coordinates loading, `generator/generator.js` owns generation sessions, and `execution/layer.js` executes layers. |

Developer guides are operational checklists, not normative contracts; contracts
stay in `docs/style/*.md`, `docs/config.md`, `docs/conversion-runtime-contract.md`
and the other core contract docs. Protocol documents under `docs/agents/` load on
demand through skills (index: `docs/agents/README.md`), not for every task.

A change that could silently rewrite manifest/runtime behavior must fail fast
with an actionable error or add a regression test proving it cannot. Keep component support, artifact preparation
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
│   ├── experimental/     # Quarantined internal-only lanes: adapters, audio, bdpa, bridge, browser, diffusion, distribution, embeddings, energy, hotswap, loader, logits, orchestration, training, turboquant
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

CATSCAN.md files record a component's owner, contracts and invariants. Look up
the one for the component you change when you need its contract; you don't
have to read the chain before editing. When code and a charter disagree, decide
which is right: fix the code if it broke the contract, or move the boundary and
update the charter in the same commit. Never rewrite a charter to excuse a
failure.

When you change a charter, component boundary, or the CATSCAN policy inventory,
run `npm run catscan:check`.

### Public vs Internal Tooling

Full breakdown: `docs/agents/tooling-surface.md`.

- **Public CLI**: `src/cli/doppler-cli.js` — ships as `bin.doppler`. The package's other bins are `dopplerx` and `doppler-gpu` (both `src/cli/doppler-quickstart.js`, the quickstart) and `doppler-serve` (`src/cli/doppler-serve.js`, the OpenAI-compatible localhost server).
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
wall is classified (GPU compute versus submit/readback/orchestration) to target
the cause of a slowdown. Measure the changed path early and iterate.

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
tokens, silent quality loss), trace the changed path to its first broken
boundary, repair it, and rerun the affected workload. The `doppler-debug` skill
and its [regression steps](docs/agents/debug-protocol.md#8-regression-after-a-code-change)
are available when they help the investigation.
Workarounds are not fixes: label a disabled broken path as a workaround, track
the underlying bug, and add a comment or test stating what it avoids and why.

### Logging

Use the debug module (`src/debug/index.js`), not raw `console.*`, in runtime code. Direct console output is allowed in `src/cli/` for the requested command result or diagnostic output, in `src/debug/` (the log sink), `demo/`, `tools/`, `tests/` and `benchmarks/`, and for one-time startup in `src/gpu/device.js`; `source:style:check` enforces the `src/`/`demo/` part.

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
