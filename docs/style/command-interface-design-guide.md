# DOPPLER Command Interface Design Guide

Design rules for the command interface, independent of surface.
This applies to browser clients and the Node CLI equally.

---

## Goals

- **One contract**: browser and CLI use the same command schema.
- **Config-only control**: command behavior is encoded in runtime config.
- **Shared fairness contract**: cross-engine benchmark axes are defined once and applied to all engines.
- **Explicit exits**: each command has a measurable completion condition.
- **Deterministic intent mapping**: command -> intent cluster is fixed.
- **Parity by default**: no browser-only or CLI-only command semantics.

## Deterministic Command Mapping

- The only supported mapping is:
  - `convert` → convert intent
  - `debug` → investigate intent
  - `bench` → calibrate intent
  - `verify` → verify intent
  - `lora` → operator lifecycle (no harness intent)
  - `distill` → operator lifecycle (no harness intent)
- Mapping lives in JSON-first runtime command metadata and `normalizeToolingCommandRequest()`.
- Any unknown command, unsupported surface combination, or missing required contract value must fail fast.
- No command surface may substitute behavior when capabilities differ; failures must surface at surface boundary.

---

## Canonical Commands

- `convert`
- `debug`
- `bench`
- `verify`
- `lora`
- `distill`

Defined in `src/tooling/command-api.js`.
All surfaces must normalize via `normalizeToolingCommandRequest()`.

---

## Intent Clusters

### Verification (Gatekeeper)

- **Intent**: `verify`
- **Exit**: pass/fail suite summary + diagnostics
- **Command**: `verify`

### Investigation (Microscope)

- **Intent**: `investigate`
- **Exit**: trace/profile/log artifacts or interactive output
- **Command**: `debug`

### Calibration (Yardstick)

- **Intent**: `calibrate`
- **Exit**: comparable scalar metrics
- **Command**: `bench`
- Training calibration runs through `bench` with `workloadType="training"` and
  must remain behaviorally distinct from verify-path `verify` with
  `request.suite="training"` in `--config`.
- Training payloads are schema-pinned (`trainingSchemaVersion=1` for training flows).
- Force-resume audit fields are explicit and fail-closed:
  - `forceResumeReason`, `forceResumeSource`, and `checkpointOperator` require
    `forceResume=true`.
- Diffusion calibration runs through `bench` with `workloadType="diffusion"` and
  must remain behaviorally distinct from verify-path `verify` with
  `request.suite="diffusion"` in `--config`.
- Diffusion verify/calibrate results must include
  `metrics.performanceArtifact` with required stage lanes:
  `cpu.prefillMs`, `cpu.denoiseMs`, `cpu.vaeMs`, and `cpu.totalMs`.

### Maintenance

- **Intent**: none (unless harnessed workload is executed)
- **Exit**: materialized artifact + hashes
- **Command**: `convert`

### Training Operator Lifecycle

- **Intent**: none (operator lifecycle is workload-driven, not harness-driven)
- **Exit**: workload-locked run root plus checkpoint/eval/scoreboard/compare/quality-gate artifacts
- **Commands**: `lora`, `distill`
- These commands normalize through the same command API but do not inject `runtime.shared.harness.*`.
- `lora.action` is:
  `run|eval|watch|export|compare|quality-gate|activate`
- `distill.action` is:
  `run|stage-a|stage-b|eval|watch|compare|quality-gate|subsets`
- Operator commands are workload-first:
  `workloadPath` or `runRoot` is required, and behavior-changing eval/subset/checkpoint policy must come from workload JSON or explicit artifact references.

---

## Runtime Contract

For harnessed commands (`debug`, `bench`, `verify`), runners must apply:

- `runtime.shared.harness.mode`
- `runtime.shared.harness.modelId` (required except `kernels` suite)
- `runtime.shared.tooling.intent`

`verify` verify suites are: `kernels`, `inference`, `training`, `diffusion`, and `energy`.

Diffusion command contracts:
- Verify path: `command="verify"` and `suite="diffusion"`.
- Calibrate path: `command="bench"`, `suite="bench"`, and `workloadType="diffusion"`.
- Runtime backend contract: `runtime.inference.diffusion.backend.pipeline="gpu"` only.
- Both paths must emit timing diagnostics and diffusion stage metrics as contract artifacts.

Use `buildRuntimeContractPatch()` and merge into runtime config before execution.

For cross-engine benchmarks, maintain a two-layer contract:

1. Shared benchmark contract (prompt/workload/sampling/run policy)
2. Engine overlay (engine-specific execution knobs)

The shared contract is part of command semantics; engine overlays must not mutate fairness axes.

Commands are rejected when:

- command is unknown
- required suite/model fields are missing
- suite/intent contract is violated
- `calibrate` enables investigation instrumentation
- benchmark shared-contract fields drift across compared engines
- operator command action is unknown
- operator command is missing `workloadPath`/`runRoot` or required checkpoint/subset references

---

## Surface Parity Rules

- Every command must run through `ensureCommandSupportedOnSurface()`.
- Surface capability limits are explicit failures, not alternate behavior.
- Output envelope shape is stable across surfaces:
  - Success: `{ ok: true, schemaVersion: 1, surface, request, result }`
  - Error: `{ ok: false, schemaVersion: 1, surface|null, request|null, error: { code, message, details, retryable } }`
- New command capability is valid only when:
  1. Added to `src/tooling/command-api.js`
  2. Implemented in both browser and Node runners
  3. Documented in this guide and harness guide

### CLI Surface Selection

- CLI supports `--surface auto|node|browser`.
- `--surface auto` is explicit transport resolution for harnessed commands: try Node first, then browser relay only when Node WebGPU is unavailable. Command intent and contract stay unchanged.
- Exception: training flows (`suite="training"` or `bench + workloadType="training"`) must not auto-downgrade from node to browser; this is a fail-closed transport rule that preserves command semantics.
- `lora` and `distill` are also training flows for auto-surface purposes and must not silently downgrade.
- Node runner may bootstrap WebGPU from available runtime support before failing.
- Browser relay executes `runBrowserCommand()` in a browser via `src/tooling/command-runner.html`
  (default headless, configured via `run.browser` fields in CLI `--config`).
- Browser relay can attach to an existing server with `run.browser.baseUrl`.
- `convert` is Node-only in CLI (`--surface browser` is rejected).
- `lora` and `distill` are currently Node-only and must fail closed on browser surfaces until equivalent runtime semantics exist there.
- `keepPipeline=true` is rejected on browser relay because pipeline objects are not serializable across process boundaries.
- `convert` execution tuning belongs in `request.convertPayload.execution` and must not change command semantics.

---

## References

- `src/tooling/command-api.js`
- `src/tooling/browser-command-runner.js`
- `src/tooling/node-command-runner.js`
- `src/tooling/node-browser-command-runner.js`
- `docs/style/harness-style-guide.md`
- `docs/style/config-style-guide.md`
