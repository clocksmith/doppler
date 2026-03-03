# DOPPLER Harness Interface Style Guide

Design rules for command execution across browser harnesses and the Node CLI.

---

## Goals

- **Shared contract**: one command schema for all execution surfaces.
- **Config-first control**: tunables live in runtime config.
- **Reproducibility**: same config + model + inputs -> same run behavior.
- **Explicit capability gates**: unsupported environment features fail fast.

---

## Vocabulary

- **Command**: `convert`, `debug`, `bench`, `verify`
- **Suite**: `kernels`, `inference`, `training`, `bench`, `debug`, `diffusion`, `energy`
- **Runtime contract patch**: `{ shared.harness.mode, shared.harness.modelId, shared.tooling.intent }`
- **Surface**: `browser` or `node`

---

## Execution Rules

- Normalize requests with `normalizeToolingCommandRequest()`.
- Validate surface support with `ensureCommandSupportedOnSurface()`.
- For harnessed runs, apply `buildRuntimeContractPatch()` before execution.
- Do not mutate command semantics per surface.
- Do not add hidden model/mode/intent defaults in UI or CLI wrappers.

---

## Browser Rules

- Browser UI may compose runtime config for interactive workflows.
- Browser runner entrypoint: `runBrowserCommand()`.
- Browser conversion must use the `convert` command and explicit payload.
- Benchmark suites may compose benchmark-level runtime fields from
  `runtime.shared.benchmark.run` (for example `customPrompt`, `maxNewTokens`,
  `sampling`, and run counts).

## Node Rules

- Node runner entrypoint: `runNodeCommand()`.
- Node CLI entrypoint: `tools/doppler-cli.js`.
- Node suite runs require WebGPU support. Provider resolution order is:
  `DOPPLER_NODE_WEBGPU_MODULE` (explicit override) -> local sibling `../fawn/nursery/webgpu-core` when present
  -> `@doe/webgpu-core` -> optional `webgpu`; if none resolve, fail explicitly.

---

## URL + Preset Contract

Browser harness URLs accept only:

- `runtimePreset`
- `runtimeConfig`
- `runtimeConfigUrl`
- `configChain`

CLI equivalents:

- `--config` (canonical command payload)
- `--runtime-config` (optional runtime override: JSON, URL, or file path)

No per-field tunable flags are allowed outside runtime config.
For benchmarked commands, shared-contract values (prompt/workload/sampling/run
policy) must be provided through
`runtimeConfig`/`runtimePreset`/`runtimeConfigUrl` and materialized as
`runtime.shared.benchmark.run`.

---

## Logging and Outputs

- Runtime code uses debug module (`src/debug/*`).
- Entrypoints may print status and progress to console.
- Result envelopes must keep a stable schema:
  - Success: `{ ok: true, schemaVersion: 1, surface, request, result }`
  - Error: `{ ok: false, schemaVersion: 1, surface|null, request|null, error: { code, message, details, retryable } }`

---

## See Also

- `docs/style/command-interface-design-guide.md`
- `docs/style/config-style-guide.md`
- `docs/style/benchmark-style-guide.md`
