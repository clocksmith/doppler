# Doppler Test Harness

## Overview

This directory contains browser harnesses for Doppler inference and GPU kernels.
The same command contract is available through Node CLI and browser relay.

## Unified Harness (Browser)

**URL:** `http://localhost:8080/tests/harness.html` (serve repo root with a static server)

Modes are configured in `runtime.shared.harness` and passed through runtime config.
The harness does not accept per-field query overrides.

| Mode | Purpose |
|------|---------|
| `kernels` | GPU kernel correctness tests |
| `inference` | Inference pipeline tests |
| `bench` | Benchmark runner shell |
| `training` | Training kernel tests |
| `energy` | Energy model runs |

## Unified Commands (CLI, 1:1 schema)

```bash
npm run verify:model -- --config '{"request":{"suite":"kernels"},"run":{"surface":"auto"}}'
npm run verify:model -- --config '{"request":{"suite":"inference","modelId":"gemma-3-1b-q4"},"run":{"surface":"auto"}}'
npm run verify:model -- --config '{"request":{"suite":"training","modelId":"gemma-3-1b-q4"},"run":{"surface":"auto"}}'
npm run debug -- --config '{"request":{"modelId":"gemma-3-1b-q4","runtimePreset":"modes/debug"},"run":{"surface":"auto"}}'
npm run bench -- --config '{"request":{"modelId":"gemma-3-1b-q4","runtimePreset":"experiments/bench/gemma3-bench-q4k"},"run":{"surface":"auto"}}'
```

Surface behavior:

- `--surface auto` uses Node first and falls back to browser relay if Node WebGPU is unavailable.
- `--surface node` requires a WebGPU-enabled Node runtime.
- `--surface browser` runs headless browser harness through `src/tooling/command-runner.html`.

## Browser Relay Config

Browser relay options are configured under `run.browser` in `--config`
(for example `channel`, `executablePath`, `headless`, `port`, `timeoutMs`,
`runnerPath`, `staticRootDir`, `baseUrl`, `browserArgs`, and `console`).

## Running Browser Harness Manually

```bash
python3 -m http.server 8080
```

Example runtime config payload:

```json
{"shared":{"harness":{"mode":"inference","autorun":true,"skipLoad":false,"modelId":"gemma3-1b-q4"}}}
```

## Shared Utilities

`src/inference/test-harness.js` exposes shared helpers:

- `discoverModels`
- `parseRuntimeOverridesFromURL`
- `createHttpShardLoader`
- `fetchManifest`
- `initializeDevice`
- `createTestState`

## Related

- `../docs/architecture.md`
- `../docs/testing.md`
- `../docs/style/command-interface-design-guide.md`
