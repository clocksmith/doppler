# Doppler CLI Quick Start

This file is a command-shape reference for the CLI surface.
For the canonical day-1 workflow (verify -> optional convert -> bench), use
[getting-started.md](getting-started.md).

## Entrypoint

```bash
node tools/doppler-cli.js
```

## Command contract

| Command | Required request fields | Notes |
| --- | --- | --- |
| `convert` | `request.inputDir`, `request.convertPayload.converterConfig` | Node surface only |
| `verify` | `request.suite` (+ `request.modelId` except `kernels`) | Emits pass/fail contract output |
| `debug` | `request.modelId` (or `request.modelUrl`) | Investigation intent |
| `bench` | `request.modelId` (or `request.modelUrl`) | Calibration intent |

Supported surfaces: `--surface auto|node|browser`.

## Minimal examples

```bash
node tools/doppler-cli.js verify --config '{"request":{"suite":"inference","modelId":"gemma-3-270m-it-wq4k-ef16-hf16"},"run":{"surface":"auto"}}' --json

node tools/doppler-cli.js debug --config '{"request":{"modelId":"gemma-3-270m-it-wq4k-ef16-hf16","runtimePreset":"modes/debug"},"run":{"surface":"auto"}}' --json

node tools/doppler-cli.js bench --config '{"request":{"modelId":"gemma-3-270m-it-wq4k-ef16-hf16"},"run":{"surface":"auto","bench":{"save":true,"saveDir":"benchmarks/vendors/results"}}}' --json
```

## Conversion notes

- `convert` does not take `modelId`; set `output.modelBaseId` in converter config.
- `loadMode=memory` is Node-only and requires local filesystem model data.
- Prefer immutable Hugging Face revisions for reproducible runs.

For conversion config details, see
[../tools/configs/conversion/README.md](../tools/configs/conversion/README.md).
