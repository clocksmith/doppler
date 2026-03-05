# DOPPLER Getting Started

This is the canonical day-1 workflow for Doppler.

## Scope

Use this guide for:
- first successful `verify`
- optional local conversion
- first benchmark artifact

For hardware sizing and expected performance, see [performance-sizing.md](performance-sizing.md).

## Prerequisites

- Node.js 20+
- repo dependencies installed
- WebGPU-capable runtime for `verify`, `debug`, and `bench`

CLI entrypoint:

```bash
node tools/doppler-cli.js
```

## Path A: Run a prebuilt RDRR model

Use this when a model is already in the hosted registry.

```bash
HF_REVISION=4efe64a914892e98be50842aeb16c3b648cc68a5
MODEL_ID=gemma-3-270m-it-wq4k-ef16-hf16
MODEL_URL="https://huggingface.co/Clocksmith/rdrr/resolve/${HF_REVISION}/models/gemma-3-270m-it-wq4k-ef16"
```

### Verify

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

### Benchmark

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

Use this when no prebuilt RDRR artifact exists.

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

Conversion writes artifacts to a filesystem output directory, not into the
browser shard-manager store. To verify, run via browser relay so the model
loads through the full OPFS-to-VRAM pipeline:

```bash
MODEL_ID=$(node -e "const fs=require('fs');const j=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));console.log(j.output.modelBaseId);" "${CONVERSION_CONFIG}")
OUTPUT_DIR=$(node -e "const fs=require('fs');const j=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));console.log(j.output.outputDir || 'models/local');" "${CONVERSION_CONFIG}")

node tools/doppler-cli.js verify --config "{
  \"request\": {
    \"suite\": \"inference\",
    \"modelId\": \"${MODEL_ID}\",
    \"modelUrl\": \"file://${OUTPUT_DIR}\",
    \"loadMode\": \"http\",
    \"cacheMode\": \"warm\",
    \"runtimePreset\": \"modes/debug\"
  },
  \"run\": { \"surface\": \"browser\" }
}" --json
```

Note: `surface: "browser"` is required here. The converted artifacts live on
the filesystem, not in shard-manager storage. Using `surface: "auto"` or
omitting `modelUrl`/`loadMode` will fail because the verify harness expects
the model in OPFS, which the conversion step does not populate.

## Next docs

- CLI command reference patterns: [cli-quickstart.md](cli-quickstart.md)
- Setup and environment troubleshooting: [setup-instructions.md](setup-instructions.md)
- Onboarding consistency checks and scaffolders: [onboarding-tooling.md](onboarding-tooling.md)
- Benchmark policy and claims: [benchmark-methodology.md](benchmark-methodology.md)
