# Model Catalog

## Registry And Storage

### Repo `models/catalog.json` (model registry source)

`models/catalog.json` is the repo source of truth for supported model metadata:
model IDs, labels, aliases, lifecycle, artifact identity, quickstart/demo
visibility, vendor benchmark mapping, and Hugging Face hosted coordinates.

Generated mirrors include:

- `src/client/doppler-registry.json`
- `docs/model-support-matrix.md`
- `docs/model-support-inventory.md`
- HF `Clocksmith/rdrr` `registry/catalog.json`

Use `npm run ci:catalog:check` after catalog or hosted registry changes.

### Repo `models/local/` (developer-local artifact cache)

Path: `models/local/<model-id>/`

This is a developer-local cache for manifests and tokenizer sidecars. It is not
a release or CI source of truth. Git tracks everything except shards:

- `manifest.json` — full RDRR manifest (inference config, execution graph, architecture, tensor layout)
- `origin.json` — conversion provenance (source repo, revision, format, timestamp)
- `tokenizer.json` / `tokenizer.model` — tokenizer assets

Shards (`shard_*.bin`) may also be present on disk (copied from the external volume or HF). They are gitignored.

### External volume (complete copies)

Path: `$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<model-id>/`
Env: `DOPPLER_EXTERNAL_MODELS_ROOT` (auto-detected; `/Volumes/models` or `/Volumes/models2` on macOS, `/media/x/models` on Linux)

The external volume owns complete artifact bytes: manifests, tokenizers, and
shards. Sync manifests and sidecars intentionally; do not infer catalog state
from a local directory listing.

Supporting directories:
- `huggingface_cache/` — HF Hub cache for source checkpoints (SafeTensors/GGUF)

### Hugging Face (hosted artifacts)

Repo: `Clocksmith/rdrr`
Path: `models/<model-id>` within the repo

The verified, promotion-ready subset is published here. See [registry-workflow.md](../docs/registry-workflow.md).

### Workflow

1. Convert model → output lands on external drive (shards + manifest + origin + tokenizer)
2. Copy non-shard files to `models/local/<model-id>/`
3. Verify the artifact and update `models/catalog.json` only after the output is accepted for support
4. Run support matrix/inventory sync checks
5. Publish: tool reads catalog metadata plus artifact files, assembles and uploads

## `catalog.json` Schema

Each entry supports:

- `modelId` (string, required)
- `label` (string, optional)
- `description` (string, optional)
- `mode` or `modes` (`run`, `embedding`, or other explicit surface labels used by tooling)
- `baseUrl` (string or null). `null` for models that are HF-only or external-volume-only. Non-null only for models served from a repo-relative path in demo deployments.
- `sizeBytes` (number, optional)
- `recommended` (boolean, optional)
- `sortOrder` (number, optional)
- `demoVisible` (boolean, optional): `true` surfaces a hosted model in the web demo without adding it to quickstart; `false` hides a quickstart model from the web demo.
- `demoWarningBadges` (string array, optional): compact warning badges for demo model cards.
- `demoWarningText` (string, optional): short warning text for demo model cards.
- `hf` (object, optional): `repoId`, `revision`, `path` for Hugging Face hosted artifacts
- `lifecycle` (object, optional but recommended)
  - `availability` (object): `curated` | `local` | `hf` booleans
  - `status` (object): `runtime`, `conversion`, `demo`, `tested` labels
  - `tested` (object): latest verification metadata
    - `suite` (string)
    - `surface` (string)
    - `result` (`pass` | `fail` | `unknown`)
    - `lastVerifiedAt` (`YYYY-MM-DD`)
    - `source` (string, e.g. `registry-verify`)
    - `contracts` (object): `executionContractOk`

## Related Docs

- [model-promotion-playbook.md](../docs/model-promotion-playbook.md) — end-to-end promotion workflow
- [registry-workflow.md](../docs/registry-workflow.md) — HF publication and catalog validation
- [config-source-of-truth.md](../docs/developer-guides/config-source-of-truth.md) — layered ownership map
- [conversion-runtime-contract.md](../docs/conversion-runtime-contract.md) — conversion-static vs runtime-overridable fields
