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
- `weightsLayout`
- `tensors`
- `metadata`

Schema reference:
- `src/experimental/adapters/adapter-manifest.js`

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

Matrix layout is explicit in [layout policy](../src/config/lora-layouts.json).
Native `input-major` exports store A as `[input, rank]` and B as
`[rank, output]`. Standard PEFT stores A as `[rank, input]` and B as
`[output, rank]`; raw imports must declare `weightsLayout: "peft"` in the
manifest or load options. Signed `peft_safetensors` requests select that layout
from their format and reject a conflicting manifest. WGSL reads the original
matrix orientation; JavaScript does not transpose tensor data during inference.
The loader retains the declared shapes, validates the rank axes, and binds
PEFT layout into execution identity. Existing raw manifests without a layout
retain the configured native layout and their prior execution identity.

## Export path

Use training export helper:

```js
import { exportLoRAAdapter } from '../src/experimental/training/export.js';
```

Runtime loader paths:
- `src/experimental/adapters/lora-loader.js`
- `src/experimental/adapters/adapter-manifest.js`
- `src/experimental/training/export.js`

Adapter IDs, source-weight SHA-256 identities, and resolved execution digests
are checked against the bundled deny-only registry before activation. See
[`revocation.md`](revocation.md).

## Interop note

GGUF conversion is external to this repo flow.
