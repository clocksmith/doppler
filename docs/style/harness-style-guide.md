# Doppler Harness Interface Style Guide

Harness-specific rules for browser and Node execution surfaces.

Command semantics, intent mapping, and envelope contracts are canonical in
[command-interface-design-guide.md](./command-interface-design-guide.md).

## Scope

Harness URL/runtime-config boundaries, browser relay and Node runner behavior,
and harness-only constraints that are not command-level semantics.

## Harness responsibilities

- Apply runtime config without changing command meaning.
- Preserve surface parity: no hidden defaults that change outcomes.
- Fail fast on unsupported environment capabilities.
- Keep runtime tuning config-driven.

## Browser harness contract

Browser harness URLs accept only `runtimeProfile`, `runtimeConfig`,
`runtimeConfigUrl`, and `configChain`; no per-field URL tuning.

Browser runner entrypoint: `runBrowserCommand()`.

## Node harness contract

Node runner entrypoint: `runNodeCommand()`. CLI entrypoint: `src/cli/doppler-cli.js`.

Node WebGPU provider resolution order:
- `DOPPLER_NODE_WEBGPU_MODULE` if set (fail-closed: when set, Doppler does not continue to later providers)
- otherwise pre-installed WebGPU when `navigator.gpu.requestAdapter` and the core GPU enums are present, and adapter probing succeeds
- otherwise `webgpu`
- otherwise fail explicitly

## Runtime input composition

Apply, in order: `configChain` (when supported), `runtimeProfile`,
`runtimeConfigUrl`, `runtimeConfig`. Do not short-circuit after the first
provided input or silently ignore a supported field on one surface. Command
metadata stays outside runtime config, in the explicit request/suite context.

## Explicit Source Selection

- When a caller provides an explicit `modelUrl` or source URL, manifest/source comparison failures must fail closed; never silently reuse cached artifacts.
- If explicit source selection cannot be honored, surface an actionable error at the load boundary.
- Browser relay may rewrite an explicit local `file://` model directory onto a relay-owned same-origin static mount.
- When `run.browser.baseUrl` points at an external server, browser relay cannot honor a local `file://` `modelUrl` and must fail closed.

## Background Side Effects

Background auto-tuning, prewarming, or other behavior-changing work requires
explicit config opt-in; never start it just because a model loaded.

## Diffusion Contract

Diffusion verification uses `workload="diffusion"` via `verify`; calibration
uses `workload="diffusion"` via `bench`. Harness metadata must include
`workloadType="diffusion"` and `suite="diffusion"`.

## Logging

- Run code uses `src/debug/*`; entrypoints may print status/progress.
- Keep output deterministic for the same config + model + workload.

## See also

- [config-style-guide.md](./config-style-guide.md)
- [benchmark-style-guide.md](./benchmark-style-guide.md)
