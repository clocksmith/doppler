# Model Catalog

## Storage Tiers

Model artifacts live across three tiers with clear ownership.

### 1. Repo `models/local/` (source of truth for manifests)

Path: `models/local/<model-id>/manifest.json`

This is the canonical location for model manifests and provenance metadata. These small JSON files are tracked in git and define the correctness contract for each model. Tests validate against these manifests. Publication to Hugging Face reads manifests from here.

Contents per model directory:
- `manifest.json` — full RDRR manifest (inference config, execution graph, architecture, tensor layout)
- `origin.json` — conversion provenance (source repo, revision, format, timestamp)

Weight shards and tokenizer files are NOT stored here (too large for git).

### 2. External volume (shard container)

Path: `$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/<model-id>/`
Env: `DOPPLER_EXTERNAL_MODELS_ROOT` (auto-detected; `/Volumes/models` on macOS, `/media/x/models` on Linux)

This is a shard container for heavy binary files. It holds `shard_*.bin`, `tokenizer.json`, and may have copies of `manifest.json` and `origin.json` from conversion output — but these are NOT the source of truth. The `models/local/` manifests take precedence.

The publish tool assembles uploads from `models/local/` manifests + external drive shards.

Supporting directories:
- `huggingface_cache/` — HF Hub cache for source checkpoints (SafeTensors/GGUF)

### 3. Hugging Face (hosted artifacts for browser/demo)

Repo: `Clocksmith/rdrr`
Path: `models/<model-id>` within the repo

Only the verified, promotion-ready subset is published here. The remote registry is rebuilt from the approved hosted set on each publish. See [registry-workflow.md](../docs/registry-workflow.md).

### Workflow

1. Convert model → output lands on external drive (shards + manifest + origin)
2. Copy `manifest.json` and `origin.json` to `models/local/<model-id>/`
3. Run tests against `models/local/` manifests
4. Publish: tool reads manifest from `models/local/`, shards from external drive, assembles and uploads

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
- [conversion-runtime-contract.md](../docs/conversion-runtime-contract.md) — conversion-static vs runtime-overridable fields
