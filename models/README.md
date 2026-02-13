# Quick Model Catalog

Use `catalog.json` to define hosted models that the demo can import directly into OPFS.

Model storage under this directory is split into two buckets:

- `models/curated/**`: deployable, user-facing quick-download models
- `models/local/**`: local development/testing models only (never deployed)

Each entry supports:

- `modelId` (string, required)
- `label` (string, optional)
- `description` (string, optional)
- `mode` or `modes` (`text`, `embedding`, `both`; optional, defaults to `text`)
- `baseUrl` (string, optional). If omitted, defaults to `./curated/<modelId>` relative to `catalog.json`.
- `sizeBytes` (number, optional)
- `recommended` (boolean, optional)
- `sortOrder` (number, optional)

Recommended layout for deployable assets:

- `models/catalog.json`
- `models/curated/<model-id>/manifest.json`
- `models/curated/<model-id>/shard_*.bin`
- `models/curated/<model-id>/tokenizer.json` (if bundled)
- `models/curated/<model-id>/tokenizer.model` (if sentencepiece)

Recommended layout for local-only testing assets:

- `models/local/<model-id>/manifest.json`
- `models/local/<model-id>/shard_*.bin`
- `models/local/<model-id>/tokenizer.json` (if bundled)
- `models/local/<model-id>/tokenizer.model` (if sentencepiece)

Notes:

- `models/local/**` is allowed by the layout validator for local testing.
- `models/local/**` is ignored by Firebase hosting for the Doppler deploy target.
- Keep `catalog.json` entries pointed at curated paths when you want models downloadable in production.

Example `catalog.json` entry:

```json
{
  "modelId": "example-embedding-model",
  "label": "Example Embedding",
  "mode": "embedding",
  "baseUrl": "./curated/example-embedding-model",
  "sizeBytes": 123456789,
  "recommended": true,
  "sortOrder": 10
}
```
