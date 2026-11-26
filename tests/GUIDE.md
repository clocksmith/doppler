# Doppler Test Harness

## Overview

This directory contains browser harnesses for Doppler inference and GPU kernels.
The same command contract is available through Node CLI and browser relay.

## Unified Harness (Browser)

**URL:** `http://localhost:8080/tests/harness.html` (serve the repo root with
`node tools/serve-static.js --port 8080`)

Runtime inputs stay runtime-only on the harness page:
- `runtimeProfile`
- `runtimeConfig`
- `runtimeConfigUrl`
- `configChain`

Command/run context is separate from runtime config. When driving `tests/harness.html`
manually, pass `mode`, `workload`, `modelId`, `autorun`, and `skipLoad` as page
context instead of encoding them under `runtime.shared.harness` or
`runtime.shared.tooling`.

The page dispatches on `workload` (default `kernels`); `mode` defaults to
`verify` and is only a label, except `mode=simulation`, which selects the
simulation workload.

| `workload` | Purpose |
|------|---------|
| `kernels` | GPU kernel correctness tests |
| `inference` | Inference pipeline tests (needs `modelId`) |
| `bench` | Benchmark runner shell |
| `training` | Training kernel tests |
| `energy` | Energy model runs (needs `modelId`) |

Passing `manifestUrl` instead runs a harness manifest through
`runBrowserManifest` from `src/tooling/browser-harness.js`.

## Unified Commands (CLI, 1:1 schema)

```bash
npm run verify:model -- --config '{"request":{"workload":"kernels"},"run":{"surface":"auto"}}'
npm run verify:model -- --config '{"request":{"workload":"inference","modelId":"gemma-3-270m-it-q4k-ehf16-af32"},"run":{"surface":"auto"}}'
npm run verify:model -- --config '{"request":{"workload":"training","modelId":"gemma-3-270m-it-q4k-ehf16-af32"},"run":{"surface":"auto"}}'
npm run debug -- --config '{"request":{"modelId":"gemma-3-270m-it-q4k-ehf16-af32","runtimeProfile":"profiles/verbose-trace"},"run":{"surface":"auto"}}'
npm run bench -- --config '{"request":{"modelId":"gemma-3-270m-it-q4k-ehf16-af32","runtimeProfile":"experiments/bench/gemma3-bench-q4k"},"run":{"surface":"auto"}}'
```

Surface behavior:

- `--surface auto` uses Node first and falls back to browser relay only for harness-compatible commands when Node WebGPU is unavailable.
- Training verify flows and operator commands (`lora`, `distill`) are fail-closed auto-surface exceptions and do not downgrade to browser relay.
- `--surface node` requires a WebGPU-enabled Node runtime.
- `--surface browser` runs headless browser harness through `src/tooling/command-runner.html`.

## Browser Relay Config

Browser relay options are configured under `run.browser` in `--config`
(for example `channel`, `executablePath`, `headless`, `port`, `timeoutMs`,
`runnerPath`, `staticRootDir`, `baseUrl`, `browserArgs`, and `console`).

## Running Browser Harness Manually

```bash
node tools/serve-static.js --port 8080
```

See also [Browser harness](../docs/testing-runbook.md#browser-harness).

Example harness URL:

```text
http://localhost:8080/tests/harness.html?mode=verify&workload=inference&modelId=gemma-3-270m-it-q4k-ehf16-af32&autorun=true
```

## Shared Utilities

`src/tooling/test-harness.js` exposes shared helpers:

- `discoverModels`
- `parseRuntimeOverridesFromURL`
- `createHarnessShardStorageContext` (shard storage for a model URL and manifest)
- `initializeInference` (device, manifest, storage and pipeline in one call)
- `createTestState`

Device setup uses `initDevice` and `getKernelCapabilities` from
`src/gpu/device.js`; manifests are parsed with `parseManifest` from
`src/formats/rdrr/index.js`.

## Related

- `../docs/architecture.md`
- `../docs/testing.md`
- `../docs/style/command-interface-design-guide.md`
