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
npm run test:model -- --suite kernels --surface auto
npm run test:model -- --suite inference --model-id gemma-3-1b-q4 --surface auto
npm run debug -- --model-id gemma-3-1b-q4 --runtime-preset modes/debug --surface auto
npm run bench -- --model-id gemma-3-1b-q4 --runtime-preset experiments/gemma3-bench-q4k --surface auto
```

Surface behavior:

- `--surface auto` uses Node first and falls back to browser relay if Node WebGPU is unavailable.
- `--surface node` requires a WebGPU-enabled Node runtime.
- `--surface browser` runs headless browser harness through `src/tooling/command-runner.html`.

## Browser Relay Flags

```bash
--browser-channel chrome
--browser-executable /path/to/chrome
--browser-headless true
--browser-port 8080
--browser-timeout-ms 180000
--browser-url-path /src/tooling/command-runner.html
--browser-static-root /abs/path/to/doppler
--browser-base-url http://127.0.0.1:8080
--browser-console
```

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

- `docs/architecture.md`
- `docs/testing.md`
- `docs/style/command-interface-design-guide.md`
