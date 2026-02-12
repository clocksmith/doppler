# DOPPLER Command Interface Design Guide

Design rules for the command interface, independent of surface.
This applies to browser clients and the Node CLI equally.

---

## Goals

- **One contract**: browser and CLI use the same command schema.
- **Config-only control**: command behavior is encoded in runtime config.
- **Explicit exits**: each command has a measurable completion condition.
- **Deterministic intent mapping**: command -> intent cluster is fixed.
- **Parity by default**: no browser-only or CLI-only command semantics.

---

## Canonical Commands

- `convert`
- `debug`
- `bench`
- `test-model`

Defined in `src/tooling/command-api.js`.
All surfaces must normalize via `normalizeToolingCommandRequest()`.

---

## Intent Clusters

### Verification (Gatekeeper)

- **Intent**: `verify`
- **Exit**: pass/fail suite summary + diagnostics
- **Command**: `test-model`

### Investigation (Microscope)

- **Intent**: `investigate`
- **Exit**: trace/profile/log artifacts or interactive output
- **Command**: `debug`

### Calibration (Yardstick)

- **Intent**: `calibrate`
- **Exit**: comparable scalar metrics
- **Command**: `bench`

### Maintenance

- **Intent**: none (unless harnessed workload is executed)
- **Exit**: materialized artifact + hashes
- **Command**: `convert`

---

## Runtime Contract

For harnessed commands (`debug`, `bench`, `test-model`), runners must apply:

- `runtime.shared.harness.mode`
- `runtime.shared.harness.modelId` (required except `kernels` suite)
- `runtime.shared.tooling.intent`

Use `buildRuntimeContractPatch()` and merge into runtime config before execution.

Commands are rejected when:

- command is unknown
- required suite/model fields are missing
- suite/intent contract is violated
- `calibrate` enables investigation instrumentation

---

## Surface Parity Rules

- Every command must run through `ensureCommandSupportedOnSurface()`.
- Surface capability limits are explicit failures, not alternate behavior.
- Output envelope shape is stable across surfaces:
  - `{ ok, surface, request, result }`
- New command capability is valid only when:
  1. Added to `src/tooling/command-api.js`
  2. Implemented in both browser and Node runners
  3. Documented in this guide and harness guide

### CLI Surface Selection

- CLI supports `--surface auto|node|browser`.
- `auto` first attempts Node for harnessed commands and falls back to browser relay only when Node WebGPU is unavailable.
- Browser relay executes `runBrowserCommand()` in a headless browser via `src/tooling/command-runner.html`.
- Browser relay can attach to an existing server with `--browser-base-url`.
- `convert` is Node-only in CLI (`--surface browser` is rejected).
- `keepPipeline=true` is rejected on browser relay because pipeline objects are not serializable across process boundaries.

---

## References

- `src/tooling/command-api.js`
- `src/tooling/browser-command-runner.js`
- `src/tooling/node-command-runner.js`
- `src/tooling/node-browser-command-runner.js`
- `docs/style/harness-style-guide.md`
- `docs/style/config-style-guide.md`
