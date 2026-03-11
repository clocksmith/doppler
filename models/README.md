# Quick Model Catalog

Use `catalog.json` to define the repo-visible mirror of hosted models that the demo can import directly into OPFS.
The canonical runtime/demo/hosted/tested lifecycle registry lives on the external models volume and is mirrored into `catalog.json` for repo tooling.

Model artifacts live under a single physical root:

- `models/local/**`: on-disk RDRR artifacts used for local testing and optional repo-hosted demo downloads

Lifecycle labels such as `curated`, `recommended`, and `demo` are metadata mirrored from the external support registry, not path semantics.

Each entry supports:

- `modelId` (string, required)
- `label` (string, optional)
- `description` (string, optional)
- `mode` or `modes` (`run`, `embedding`, or other explicit surface labels used by tooling)
- `baseUrl` (string, optional). Keep this explicit for downloadable demo entries; `null` is valid for non-demo/local-only catalog entries.
- `sizeBytes` (number, optional)
- `recommended` (boolean, optional)
- `sortOrder` (number, optional)
- `lifecycle` (object, optional but recommended)
  - `availability` (object): `curated` | `local` | `hf` booleans
  - `status` (object): `runtime`, `conversion`, `demo`, `tested` labels
  - `tested` (object): latest verification metadata
    - `suite` (string)
    - `surface` (string)
    - `result` (`pass` | `fail` | `unknown`)
    - `lastVerifiedAt` (`YYYY-MM-DD`)
    - `source` (string, e.g. `registry-verify`)

Recommended layout for deployable assets:

- `models/catalog.json`
- `models/local/<model-id>/manifest.json`
- `models/local/<model-id>/shard_*.bin`
- `models/local/<model-id>/tokenizer.json` (if bundled)
- `models/local/<model-id>/tokenizer.model` (if sentencepiece)

Notes:

- `models/local/**` is allowed by the layout validator for local testing.
- `models/local/**` may be served in demo/dev deployments when hosting config allows it.
- `lifecycle.availability.curated` and `lifecycle.status.demo` describe artifact status, not a separate directory tree.
- Do not rely on implicit `baseUrl` or mode defaults in docs or tooling. Catalog behavior should stay explicit and fail closed.
- Preferred ownership:
  - external canonical registry: `/media/x/models/DOPPLER_SUPPORT_REGISTRY.json`
  - repo mirror: `models/catalog.json`

Example `catalog.json` entry:

```json
{
  "modelId": "example-embedding-model",
  "label": "Example Embedding",
  "mode": "embedding",
  "baseUrl": "./local/example-embedding-model",
  "sizeBytes": 123456789,
  "recommended": true,
  "sortOrder": 10,
  "lifecycle": {
    "availability": {
      "curated": true,
      "local": true,
      "hf": true
    },
    "status": {
      "runtime": "active",
      "conversion": "ready",
      "demo": "curated",
      "tested": "verified"
    },
    "tested": {
      "suite": "inference",
      "surface": "auto",
      "result": "pass",
      "lastVerifiedAt": "2026-03-04",
      "source": "registry-verify"
    }
  }
}
```
