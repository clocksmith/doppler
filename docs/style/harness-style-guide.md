# DOPPLER Harness Interface Style Guide

Harness-specific rules for browser and Node execution surfaces.

Command semantics, intent mapping, and envelope contracts are canonical in
[command-interface-design-guide.md](./command-interface-design-guide.md).

## Scope

Use this guide for:
- harness URL/runtime-config boundaries
- browser relay and node runner behavior
- harness-only constraints that are not command-level semantics

## Harness responsibilities

- Apply runtime config without changing command meaning.
- Preserve surface parity: no hidden defaults that change outcomes.
- Fail fast on unsupported environment capabilities.
- Keep runtime tuning config-driven (`runtimePreset`, `runtimeConfig`, `configChain`).

## Browser harness contract

Browser harness URLs accept only:
- `runtimePreset`
- `runtimeConfig`
- `runtimeConfigUrl`
- `configChain`

Per-field URL tuning is not allowed.

Browser runner entrypoint: `runBrowserCommand()`.

## Node harness contract

Node runner entrypoint: `runNodeCommand()`.
CLI entrypoint: `tools/doppler-cli.js`.

Node WebGPU provider resolution order:
`DOPPLER_NODE_WEBGPU_MODULE` (explicit override) -> `@simulatte/webgpu` -> `webgpu`.
If none resolve, fail explicitly.

## Runtime patching

For harnessed runs, apply `buildRuntimeContractPatch()` before execution.
The patch fields are:
- `shared.harness.mode`
- `shared.harness.modelId` (except kernel-only flows)
- `shared.tooling.intent`

## Diffusion Contract

Diffusion verification must use `suite="diffusion"` via the `verify` command path.
Diffusion calibration must use `workloadType="diffusion"` via the `bench` command path.

## Logging

- Runtime code uses `src/debug/*`.
- Entrypoints may print status/progress.
- Keep output deterministic for the same config + model + workload.

## See also

- [command-interface-design-guide.md](./command-interface-design-guide.md)
- [config-style-guide.md](./config-style-guide.md)
- [benchmark-style-guide.md](./benchmark-style-guide.md)
