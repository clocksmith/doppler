# Doppler CLI Quick Start

This file is a command-shape reference for the CLI surface.
For the canonical day-1 workflow (verify -> optional convert -> bench), use
[getting-started.md](getting-started.md).

## Entrypoint

```bash
node tools/doppler-cli.js
```

## Command Contract

| Command | Required request fields | Notes |
| --- | --- | --- |
| `convert` | `request.inputDir`, `request.convertPayload.converterConfig` | Node surface only |
| `verify` | `request.suite` plus `request.modelId` except `kernels` | Emits pass/fail contract output |
| `debug` | `request.modelId` or `request.modelUrl` | Investigation intent |
| `bench` | `request.modelId` or `request.modelUrl` | Calibration intent |
| `distill` | `request.action` plus `request.workloadPath` or `request.runRoot` | Node-only today; browser fails closed |
| `lora` | `request.action` plus `request.workloadPath` or `request.runRoot` | Node-only today; browser fails closed |

Operator-action notes:

- `distill.watch`, `distill.compare`, `distill.quality-gate` require `runRoot`
- `distill.eval` accepts `checkpointPath` or replays finalized checkpoints already present in the run root
- `lora.watch`, `lora.compare`, `lora.quality-gate` require `runRoot`
- `lora.eval` and `lora.export` accept `checkpointPath` or re-use finalized checkpoints in the run root
- `lora.activate` is part of the command contract, but the current Node runner rejects it and points activation to the browser provider/runtime surface

Supported surfaces:

- `convert`: `--surface auto|node`
- `debug`, `bench`, `verify`: `--surface auto|node|browser`
- `lora`, `distill`: `--surface auto|node` in practice; `--surface browser` is rejected

## Minimal Examples

```bash
node tools/doppler-cli.js verify --config '{"request":{"suite":"inference","modelId":"gemma-3-270m-it-wq4k-ef16-hf16"},"run":{"surface":"auto"}}' --json

node tools/doppler-cli.js debug --config '{"request":{"modelId":"gemma-3-270m-it-wq4k-ef16-hf16","runtimePreset":"modes/debug"},"run":{"surface":"auto"}}' --json

node tools/doppler-cli.js bench --config '{"request":{"modelId":"gemma-3-270m-it-wq4k-ef16-hf16"},"run":{"surface":"auto","bench":{"save":true,"saveDir":"benchmarks/vendors/results"}}}' --json

node tools/doppler-cli.js distill --config '{"request":{"action":"subsets","workloadPath":"tools/configs/training-workloads/distill-translategemma-tiny.json"}}' --json

node tools/doppler-cli.js distill --config '{"request":{"action":"run","workloadPath":"tools/configs/training-workloads/distill-translategemma-tiny.json"}}' --json

node tools/doppler-cli.js lora --config '{"request":{"action":"run","workloadPath":"tools/configs/training-workloads/lora-toy-tiny.json"}}' --json
```

## Operator Notes

- operator runs are workload-first; prefer `workloadPath` over ad hoc request fields
- `--surface auto` for `lora` and `distill` does not downgrade to browser
- run-root artifacts live under `reports/training/<kind>/<workload-id>/<timestamp>/`
- `run_contract.json` and `workload.lock.json` are written for every operator run

## Conversion Notes

- `convert` does not take `modelId`; set `output.modelBaseId` in converter config
- `loadMode=memory` is Node-only and requires local filesystem model data
- prefer immutable Hugging Face revisions for reproducible runs

For conversion config details, see
[../tools/configs/conversion/README.md](../tools/configs/conversion/README.md).
