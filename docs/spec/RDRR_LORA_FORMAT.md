# RDRR-LoRA Adapter Format

This document defines Doppler's LoRA adapter manifest format and its on-disk layout.
It is designed to be small, self-describing, and compatible with the runtime adapter loader.

Related:
- `src/adapters/adapter-manifest.js`
- `src/adapters/lora-loader.js`
- `src/training/export.js`

## Overview

An RDRR-LoRA adapter is a JSON manifest with optional inline tensors.

- Manifest: metadata + tensor list
- Tensors: LoRA matrices `A` and `B` stored inline (base64 or array)
- Loader: maps tensor names to layer/module weights at runtime

## Manifest Schema (summary)

Required fields:
- `id`: string
- `name`: string
- `baseModel`: string (model ID the adapter was trained for)
- `rank`: integer
- `alpha`: number
- `targetModules`: array of module names

Optional fields:
- `version`, `description`
- `checksum`, `checksumAlgorithm`
- `weightsFormat`, `weightsPath`, `weightsSize`
- `tensors`: inline tensor specs
- `metadata`

See full schema in `src/adapters/adapter-manifest.js`.

## Tensor Naming

Tensor names must follow:

```
layer.{L}.{module}.lora_{a|b}
```

Examples:
- `layer.0.q_proj.lora_a`
- `layer.12.o_proj.lora_b`

Module names map through `LORA_MODULE_ALIASES` in `src/inference/pipeline/lora-types.js`.

## Inline Tensor Spec

Each tensor entry includes:

```
{
  "name": "layer.0.q_proj.lora_a",
  "shape": [inDim, rank],
  "dtype": "f32",
  "base64": "...",  // or "data": [...]
  "opfsPath": "...",
  "url": "..."
}
```

Doppler currently loads `f32` tensors for LoRA. If your source is `f16`, convert to `f32` before export.

## Adapter Export

Use the training export helper:

```
import { exportLoRAAdapter } from '../src/training/export.js';
```

This creates a manifest with inline tensors that can be loaded by `adapter-manager`.

## GGUF Interop (Optional)

RDRR-LoRA is optimized for Doppler. If you need GGUF:

1. Export the adapter to JSON (inline tensors).
2. Convert tensors to a LoRA safetensors/npz format.
3. Use llama.cpp conversion tooling to emit GGUF.

An optional helper script is provided:

```
node tools/rdrr-lora-to-gguf.js --manifest adapter.json --out ./out
```

The script emits recommended conversion steps and paths, but does not run external tools.
