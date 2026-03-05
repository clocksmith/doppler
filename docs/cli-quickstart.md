# Doppler CLI Quick Start

This file is a CLI command supplement.

For the canonical end-to-end workflow, start at [getting-started.md](getting-started.md).

## CLI entrypoint

```bash
node tools/doppler-cli.js
```

## Commands

- `convert` (Node surface only)
- `verify`
- `debug`
- `bench`

## Common command recipes

### Verify inference

```bash
node tools/doppler-cli.js verify --config '{
  "request": {
    "suite": "inference",
    "modelId": "gemma-3-270m-it-wq4k-ef16-hf16",
    "runtimePreset": "modes/debug"
  },
  "run": { "surface": "auto" }
}' --json
```

### Debug

```bash
node tools/doppler-cli.js debug --config '{
  "request": {
    "modelId": "gemma-3-270m-it-wq4k-ef16-hf16",
    "runtimePreset": "modes/debug"
  },
  "run": { "surface": "auto" }
}' --json
```

### Bench and save artifact

```bash
node tools/doppler-cli.js bench --config '{
  "request": {
    "modelId": "gemma-3-270m-it-wq4k-ef16-hf16"
  },
  "run": {
    "surface": "auto",
    "bench": { "save": true, "saveDir": "benchmarks/vendors/results" }
  }
}' --json
```

## Rules

- `convert` does not take `modelId`; set `output.modelBaseId` in conversion config.
- `debug`/`bench`/`verify` require `modelId` unless `modelUrl` is provided.
- `loadMode=memory` is Node-only and requires local filesystem source data.
- Prefer immutable HF revisions for production runs.
