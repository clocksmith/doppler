# Doppler Command Interface Design Guide

Surface-independent design rules for the command interface, applying equally
to browser clients and the Node CLI.

## Goals

- **One contract**: browser and CLI use the same command schema, with no browser-only or CLI-only command semantics.
- **Config-only control**: command behavior is encoded in runtime config.
- **Shared fairness contract**: cross-engine benchmark axes are defined once and applied to all engines.
- **Explicit exits**: each command has a measurable completion condition.
- **Deterministic intent mapping**: command -> intent cluster is fixed.
- **Explicit control-plane exception**: a command that requires local files,
  signing keys, or durable operator state is Node-only and must fail closed on
  browser surfaces without changing its semantics.

## Deterministic Command Mapping

- The only supported mapping is:
  - `convert` → convert intent
  - `refresh-integrity` → maintenance intent
  - `release` → production-release eligibility (no harness intent)
  - `debug`, `diagnose` → investigate intent
  - `bench` → calibrate intent
  - `verify` → verify intent
  - `lora`, `distill` → operator lifecycle (no harness intent)
- Mapping lives in JSON-first runtime command metadata and `normalizeToolingCommandRequest()`.
- Any unknown command, unsupported surface combination, or missing required contract value must fail fast.
- No command surface may substitute behavior when capabilities differ; failures must surface at surface boundary.

## Canonical Commands

The nine commands above are canonical, defined in `src/tooling/command-api.js`.
All surfaces must normalize via `normalizeToolingCommandRequest()`.

Maintenance/export/discovery CLI paths such as `program-bundle` and `profiles`
are intentionally outside this browser/Node command-runner contract when they
only read local files and write deterministic artifacts; they must document that
boundary and never be silently treated as harnessed browser commands.

## Workload Contract

Workload names describe the run family and are independent of command intent.

Canonical workloads: `inference`, `embedding`, `rerank`, `kernels`, `training`,
`diffusion`, `energy`.

Rules:
- `debug` and `diagnose` are valid for `inference` and `embedding`.
- `bench` is valid for `inference`, `embedding`, `training`, `diffusion`, and `energy`.
- `verify` is valid for `kernels`, `inference`, `embedding`, `rerank`, `training`, `diffusion`, and `energy`.
- `convert`, `refresh-integrity`, `release`, `lora`, and `distill` do not use workload-locked harness execution.
- `workloadType` is reserved for submodes inside a workload family, such as training stage selection or diffusion lane selection.
- `request.inferenceInput` is reserved for request-owned inference payloads such as prompt overrides and multimodal image inputs. It is workload data, not runtime tuning; sampling, kernel, and cache policy still belong in runtime config.
- Runtime profile names must be orthogonal to command and workload names: no `profiles/debug`, `profiles/bench`, or `profiles/embedding` profile IDs.

## Intent Clusters

### Verification (Gatekeeper)

- **Intent** `verify`, **command** `verify`. **Exit**: pass/fail workload summary + diagnostics.

### Investigation (Microscope)

- **Intent** `investigate`, **commands** `debug`, `diagnose`. **Exit**: trace/profile/log artifacts or interactive output.

### Calibration (Yardstick)

- **Intent** `calibrate`, **command** `bench`. **Exit**: comparable scalar metrics.
- Training calibration runs through `bench` with `workload="training"`.
  Training payloads are schema-pinned (`trainingSchemaVersion=1`).
- Force-resume audit fields are explicit and fail-closed: `forceResumeReason`,
  `forceResumeSource`, and `checkpointOperator` require `forceResume=true`.

### Maintenance

- **Intent** none (unless a harnessed workload is executed), **commands** `convert`, `refresh-integrity`. **Exit**: materialized artifact + hashes.

### Production release eligibility

- **Intent** none (the signed production-release manifest and evidence are authoritative), **command** `release`.
- **Exit**: an exact-device fleet receipt or a signed `eligible|blocked` decision bundle.
- **Phases**: `qualify|decide`; the CLI supplies `decide` for the unqualified
  `doppler release` form, while command API callers must declare the phase.
- `release` consumes only production-release, trust, signing, Capsule,
  application, and fleet-evidence inputs. Runtime config, runtime profiles,
  implicit fallback, deployment, activation, and self-promotion are forbidden.
- `qualify` requires exact target and device identity on a customer-operated
  agent. `decide` requires trusted fleet signers and verifies one receipt per
  declared target.

### Training Operator Lifecycle

- **Intent** none (workload-driven, not harness-driven), **commands** `lora`, `distill`.
- **Exit**: workload-locked run root plus checkpoint/eval/scoreboard/compare/quality-gate artifacts.
- These commands normalize through the same command API but do not rewrite runtime config with harness metadata.
- `lora.action` is `run|eval|watch|export|compare|quality-gate|activate`.
- `distill.action` is `run|stage-a|stage-b|eval|watch|compare|quality-gate|subsets`.
- Operator commands are workload-first:
  `workloadPath` or `runRoot` is required, and behavior-changing eval/subset/checkpoint policy must come from workload JSON or explicit artifact references.

## Runtime Contract

For harnessed commands (`debug`, `diagnose`, `bench`, `verify`), runners must preserve
explicit command context outside runtime config: `command`, `workload`,
normalized command intent, and `modelId` (required except `kernels` and
training-calibration flows where `bench + workload="training"` allows `modelId: null`).

Diffusion command contracts:
- Verify path: `command="verify"` and `workload="diffusion"`.
- Calibrate path: `command="bench"` and `workload="diffusion"`.
- Diffusion harness metadata: `workloadType="diffusion"` and `suite="diffusion"`.
- Runtime backend contract: `runtime.inference.diffusion.backend.pipeline="gpu"` only.
- Both paths must emit timing diagnostics and diffusion stage metrics as contract artifacts:
  `metrics.performanceArtifact` with stage lanes `cpu.prefillMs`, `cpu.denoiseMs`,
  `cpu.vaeMs`, and `cpu.totalMs`.

Runtime inputs must compose identically across Node, browser, CLI, and
harnessed manifest flows, in exactly this order: `configChain` (when supported
by the surface), `runtimeProfile`, `runtimeConfigUrl`, `runtimeConfig`.

Rules:

- `configChain` support must not exist on one harness surface and silently disappear on another.
- If a surface cannot support one of these fields, it must reject the request explicitly instead of dropping or rewriting it.
- Command metadata must remain in the request/suite context; do not rewrite `runtime.shared.*` to carry command semantics.
- CLI `--runtime-profile <id>` only sets `request.runtimeProfile` before normalization (it is not a per-field tuning flag) and must conflict with `--runtime-config`, `runtimeConfigUrl`, `runtimeConfig`, or an existing `runtimeProfile` in `--config`.

Cross-engine benchmarks keep a two-layer contract: a shared benchmark contract
(prompt/workload/sampling/run policy), which is part of command semantics, and
an engine overlay (engine-specific execution knobs) that must not mutate
fairness axes.

Commands are rejected when:

- command is unknown
- required workload/model fields are missing
- workload/intent contract is violated
- `calibrate` enables investigation instrumentation
- benchmark shared-contract fields drift across compared engines
- operator command action is unknown
- operator command is missing `workloadPath`/`runRoot` or required checkpoint/subset references

## Surface Parity Rules

- Every command must run through `ensureCommandSupportedOnSurface()`.
- CLI and browser-relay envelope shape is stable across surfaces:
  - Success: `{ ok: true, schemaVersion: 1, surface, request, result }`
  - Error: `{ ok: false, schemaVersion: 1, surface|null, request|null, error: { code, message, details, retryable } }`
- Direct runner APIs (`runNodeCommand(...)`, `runBrowserCommand(...)`) throw normalized `ToolingCommandError` values on failure; they do not return error envelopes.
- New command capability is valid only when:
  1. Added to `src/tooling/command-api.js`
  2. Implemented on every declared surface, or explicitly rejected on an
     inapplicable surface when it is a local-file/signing control-plane command
  3. Documented in this guide and harness guide

### CLI Surface Selection

- CLI supports `--surface auto|node|browser`.
- `--surface auto` is explicit transport resolution for harnessed commands: try Node first, then browser relay only when Node WebGPU is unavailable. Command intent and contract stay unchanged.
- Exception: training flows (`workload="training"`, explicit training workload variants, `lora`, and `distill`) must not auto-downgrade from node to browser; this fail-closed transport rule preserves command semantics.
- Node runner may bootstrap WebGPU from available runtime support before failing.
- Browser relay executes `runBrowserCommand()` in a browser via `src/tooling/command-runner.html`
  (default headless, configured via `run.browser` fields in CLI `--config`), and can attach to an existing server with `run.browser.baseUrl`.
- Node-only in CLI (`--surface browser` is rejected): `convert`, `refresh-integrity`,
  `release`, and `diagnose` (which also never auto-downgrades to browser relay).
  `lora` and `distill` are Node-only and fail closed on browser surfaces until
  equivalent runtime semantics exist there.
- `profiles` is CLI-local read-only discovery of checked-in runtime profile IDs: outside `normalizeToolingCommandRequest()`, never executes a harnessed workload, and accepts neither `--surface` nor `--config`.
- `keepPipeline=true` is rejected on browser relay because pipeline objects are not serializable across process boundaries.
- `convert` execution tuning belongs in `request.convertPayload.execution` and must not change command semantics.
- `refresh-integrity` rebuilds manifest integrity metadata from local artifact bytes and must not consume runtime config inputs.

## References

- `src/tooling/command-api.js`
- `src/tooling/browser-command-runner.js`
- `src/tooling/node-command-runner.js`
- `src/tooling/node-browser-command-runner.js`
- `docs/style/harness-style-guide.md`
- `docs/style/config-style-guide.md`
