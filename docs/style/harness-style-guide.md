# Doppler Harness Interface Style Guide

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
- Keep runtime tuning config-driven (`runtimeProfile`, `runtimeConfig`, `configChain`).

## Browser harness contract

Browser harness URLs accept only:
- `runtimeProfile`
- `runtimeConfig`
- `runtimeConfigUrl`
- `configChain`

Per-field URL tuning is not allowed.

Browser runner entrypoint: `runBrowserCommand()`.

## Node harness contract

Node runner entrypoint: `runNodeCommand()`.
CLI entrypoint: `tools/doppler-cli.js`.

Node WebGPU provider resolution order:
- `DOPPLER_NODE_WEBGPU_MODULE` if set
- otherwise pre-installed WebGPU when `navigator.gpu.requestAdapter` and the core GPU enums are present
- otherwise `@simulatte/webgpu`
- otherwise `webgpu`
- otherwise fail explicitly

`DOPPLER_NODE_WEBGPU_MODULE` is fail-closed: when it is set, Doppler does not continue to later providers.

## Runtime patching

For harnessed runs, apply `buildRuntimeContractPatch()` before execution.
The patch fields are:
- `shared.harness.mode`
- `shared.harness.workload`
- `shared.harness.modelId` (except kernel-only flows and training calibration via `bench + workload="training"`)
- `shared.tooling.intent`

Harness runtime-input composition must preserve shared command semantics:

- apply `configChain` first when supported
- then `runtimeProfile`
- then `runtimeConfigUrl`
- then `runtimeConfig`
- then the runtime contract patch

Do not short-circuit after the first provided input. Do not silently ignore a supported field on one surface.

## Explicit Source Selection

- When a caller provides an explicit `modelUrl` or source URL, manifest/source comparison failures must fail closed.
- Harness/provider code must not silently reuse cached artifacts when explicit source verification fails.
- If explicit source selection cannot be honored, surface an actionable error at the load boundary.

## Background Side Effects

- Background auto-tuning, prewarming, or other behavior-changing work must require explicit config opt-in.
- Harness/provider code must not start those side effects only because a model was loaded successfully.

## Diffusion Contract

Diffusion verification must use `workload="diffusion"` via the `verify` command path.
Diffusion calibration must use `workload="diffusion"` via the `bench` command path.

## Logging

- Runtime code uses `src/debug/*`.
- Entrypoints may print status/progress.
- Keep output deterministic for the same config + model + workload.

## See also

- [command-interface-design-guide.md](./command-interface-design-guide.md)
- [config-style-guide.md](./config-style-guide.md)
- [benchmark-style-guide.md](./benchmark-style-guide.md)
