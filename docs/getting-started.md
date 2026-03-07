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

## Setup

### Browser requirements

Supported:
- Chrome/Edge (recommended)
- Safari with WebGPU support
- Firefox Nightly (experimental)

Check WebGPU availability:

```javascript
const adapter = await navigator.gpu.requestAdapter();
console.log(Boolean(adapter));
```

### Browser harness and demo

Serve the repo root when you need the browser harness or demo:

```bash
python3 -m http.server 8080
```

Useful URLs:
- `http://localhost:8080/tests/harness.html`
- `http://localhost:8080/demo/`

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
browser shard-manager store. To verify a local conversion, run on the Node
surface so the command can load the `file://` artifact path directly:

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
  \"run\": { \"surface\": \"node\" }
}" --json
```

Note: `surface: "node"` is the correct local-filesystem path here. The Node
runner installs the `file://` fetch shim used by the verify/debug harnesses,
while the browser relay does not share the same local filesystem contract.
If you omit `modelUrl`/`loadMode`, the verify harness will look in persistent
storage instead of the newly converted output directory.

## Next docs

- Command contract and tooling surface: [api/tooling.md](api/tooling.md)
- Onboarding consistency checks and scaffolders: [onboarding-tooling.md](onboarding-tooling.md)
- Benchmark policy and claims: [benchmark-methodology.md](benchmark-methodology.md)
- Troubleshooting and validation workflows: [operations.md](operations.md)
