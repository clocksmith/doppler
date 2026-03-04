# Doppler CLI Quick Start

Use this for the fastest end-to-end Doppler CLI workflow.

## Prerequisites

- Node.js 20+.
- Repo dependencies installed.
- WebGPU-capable runtime for `debug` / `bench` / `verify`.
- For conversion, source weights are local (`.safetensors` directory or `.gguf` file).

CLI entrypoint:

```bash
node tools/doppler-cli.js
```

Supported commands:

- `convert` (Node surface only)
- `verify`
- `debug`
- `bench`

## Path A: Use prebuilt RDRR from Hugging Face (no conversion)

Use this when the model already exists in the Doppler RDRR registry.

```bash
HF_REVISION=4efe64a914892e98be50842aeb16c3b648cc68a5
MODEL_ID=gemma-3-270m-it-wq4k-ef16-hf16-f32
MODEL_URL="https://huggingface.co/Clocksmith/rdrr/resolve/${HF_REVISION}/models/gemma-3-270m-it-wq4k-ef16"
```

### Verify inference contract

```bash
node tools/doppler-cli.js verify --config "{
  \"request\": {
    \"suite\": \"inference\",
    \"modelId\": \"${MODEL_ID}\",
    \"modelUrl\": \"${MODEL_URL}\",
    \"loadMode\": \"http\",
    \"cacheMode\": \"warm\",
    \"runtimePreset\": \"modes/debug\"
  },
  \"run\": { \"surface\": \"auto\" }
}" --json
```

### Run debug suite

```bash
node tools/doppler-cli.js debug --config "{
  \"request\": {
    \"suite\": \"debug\",
    \"modelId\": \"${MODEL_ID}\",
    \"modelUrl\": \"${MODEL_URL}\",
    \"loadMode\": \"http\",
    \"cacheMode\": \"warm\",
    \"runtimePreset\": \"modes/debug\"
  },
  \"run\": { \"surface\": \"auto\" }
}" --json
```

### Run benchmark suite

```bash
node tools/doppler-cli.js bench --config "{
  \"request\": {
    \"modelId\": \"${MODEL_ID}\",
    \"modelUrl\": \"${MODEL_URL}\",
    \"loadMode\": \"http\",
    \"cacheMode\": \"warm\"
  },
  \"run\": {
    \"surface\": \"auto\",
    \"bench\": {
      \"save\": true,
      \"saveDir\": \"benchmarks/vendors/results\"
    }
  }
}" --json
```

## Path B: Convert locally, then verify

Use this when no prebuilt RDRR exists yet.

```bash
INPUT_PATH=/path/to/source/model
CONVERSION_CONFIG=tools/configs/conversion/embeddinggemma/embeddinggemma-300m-wq4k-ef16.json
```

### Convert

```bash
node tools/doppler-cli.js convert --config "{
  \"request\": {
    \"inputDir\": \"${INPUT_PATH}\",
    \"convertPayload\": {
      \"converterConfig\": $(cat \"${CONVERSION_CONFIG}\")
    }
  }
}"
```

### Verify converted model

```bash
MODEL_ID=$(node -e "const fs=require('fs');const j=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));console.log(j.output.modelBaseId);" "${CONVERSION_CONFIG}")

node tools/doppler-cli.js verify --config "{
  \"request\": {
    \"suite\": \"inference\",
    \"modelId\": \"${MODEL_ID}\",
    \"runtimePreset\": \"modes/debug\"
  },
  \"run\": { \"surface\": \"auto\" }
}" --json
```

## Quick rules

- `convert` does not accept `modelId`; set `convertPayload.converterConfig.output.modelBaseId` in config.
- `debug` / `bench` / `verify` require `modelId` unless `modelUrl` is provided.
- `loadMode=memory` is Node-only and expects a local filesystem source path, not HTTP URL.
- Prefer immutable HF revisions (commit SHA), not `main`, for production runs.
