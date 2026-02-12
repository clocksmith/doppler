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

- **Command**: `convert`, `debug`, `bench`, `test-model`
- **Suite**: `kernels`, `inference`, `bench`, `debug`, `diffusion`, `energy`
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

## Node Rules

- Node runner entrypoint: `runNodeCommand()`.
- Node CLI entrypoint: `tools/doppler-cli.mjs`.
- Node suite runs require WebGPU support; missing support is an explicit error.

---

## URL + Preset Contract

Browser harness URLs accept only:

- `runtimePreset`
- `runtimeConfig`
- `runtimeConfigUrl`
- `configChain`

CLI equivalents:

- `--runtime-preset`
- `--runtime-config-json`
- `--runtime-config-url`

No per-field tunable flags are allowed outside runtime config.

---

## Logging and Outputs

- Runtime code uses debug module (`src/debug/*`).
- Entrypoints may print status and progress to console.
- Result envelopes must keep `{ ok, surface, request, result }` shape.

---

## See Also

- `docs/style/command-interface-design-guide.md`
- `docs/style/config-style-guide.md`
- `docs/style/benchmark-style-guide.md`
