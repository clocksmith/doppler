# RDRR LoRA Format

Canonical specification for Doppler LoRA adapter manifests.

## Overview

An RDRR-LoRA adapter is a JSON manifest with optional inline tensors.

- manifest: metadata + tensor list
- tensors: LoRA matrices `A` and `B`
- loader: maps tensor names to runtime modules

## Required fields

- `id`
- `name`
- `baseModel`
- `rank`
- `alpha`
- `targetModules`

## Optional fields

- `version`, `description`
- `checksum`, `checksumAlgorithm`
- `weightsFormat`, `weightsPath`, `weightsSize`
- `tensors`
- `metadata`

Schema reference:
- `src/adapters/adapter-manifest.js`

## Tensor naming

Required naming pattern:

```text
layer.{L}.{module}.lora_{a|b}
```

Examples:
- `layer.0.q_proj.lora_a`
- `layer.12.o_proj.lora_b`

Module alias mapping:
- `src/inference/pipelines/text/lora-types.js`

## Inline tensor entry

```json
{
  "name": "layer.0.q_proj.lora_a",
  "shape": [128, 16],
  "dtype": "f32",
  "base64": "..."
}
```

Doppler currently loads LoRA tensors as `f32`.

## Export path

Use training export helper:

```js
import { exportLoRAAdapter } from '../src/training/export.js';
```

Runtime loader paths:
- `src/adapters/lora-loader.js`
- `src/adapters/adapter-manifest.js`
- `src/training/export.js`

## Interop note

GGUF conversion is external to this repo flow.
