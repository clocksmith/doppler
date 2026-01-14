# DOPPLER Adapter Manifest (LoRA/QLoRA)

Defines the JSON manifest format used by the LoRA/QLoRA adapter loader.
Adapters are not RDRR bundles; they use a separate schema and loader path.

Schema source: `src/adapters/adapter-manifest.js`

---

## Required Fields

```json
{
  "id": "gemma-3-1b-coding",
  "name": "Gemma 3 Coding Adapter",
  "baseModel": "gemma-3-1b",
  "rank": 16,
  "alpha": 32,
  "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "tensors": []
}
```

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Slug/ID, `[a-zA-Z0-9_-]+` |
| `name` | string | Human-readable name |
| `baseModel` | string | Base model identifier |
| `rank` | number | LoRA rank (integer) |
| `alpha` | number | LoRA alpha scaling |
| `targetModules` | string[] | Modules to modify |

---

## Optional Fields

| Field | Type | Notes |
|-------|------|-------|
| `version` | string | SemVer, default `1.0.0` |
| `description` | string | Adapter description |
| `checksum` | string | SHA-256 or BLAKE3 hash |
| `checksumAlgorithm` | string | `sha256` (default) or `blake3` |
| `weightsFormat` | string | `safetensors`, `npz`, `json`, `binary` |
| `weightsPath` | string | Path/URL to weight file |
| `weightsSize` | number | Size in bytes |
| `tensors` | array | Inline tensor specs (see below) |
| `metadata` | object | Arbitrary metadata |

---

## Tensor Entries

Inline tensors are provided as objects in `tensors`. Each tensor must include
`name` and `shape`, and must have data in one of `data`, `base64`, `opfsPath`,
or `url`.

```json
{
  "name": "layers.0.q_proj.lora_a",
  "shape": [4096, 16],
  "dtype": "f32",
  "base64": "..."
}
```

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | `layers.<idx>.<module>.lora_a|lora_b` |
| `shape` | number[] | 2D tensor shape |
| `dtype` | string | `f32`, `f16`, or `bf16` |
| `data` | number[] | Inline float data |
| `base64` | string | Base64-encoded buffer |
| `opfsPath` | string | Path in OPFS |
| `url` | string | URL to tensor data |

The loader normalizes module names using `LORA_MODULE_ALIASES` and skips
unknown tensors.

---

## Loading

Adapters are loaded via the adapter loader:

```javascript
import { loadLoRAFromUrl } from './adapters/lora-loader.js';

const adapter = await loadLoRAFromUrl('https://.../adapter.json');
```

Checksum verification runs when `checksum` is present and `skipVerify` is not
set in loader options.
