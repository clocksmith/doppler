---
name: No weights in repo tree
description: Never store model weight shards in models/local/ — external volume is source of truth for RDRR artifacts
type: feedback
---

Never store weight shards (`shard_*.bin`) in `models/local/` or anywhere in the repo tree. The external volume (`/media/x/models/rdrr/`) is the single source of truth for RDRR artifacts.

**Why:** `models/local/` accumulated ~16 GB of duplicate weights that already exist on the external volume. The repo `models/` directory should only contain catalog metadata (`catalog.json`), curated manifests, and test fixtures — never weights.

**How to apply:**
- When converting models, output to the external volume, not `models/local/`
- When tests need manifests, read from the external volume or use fixtures in `models/fixtures/`
- `catalog.json` `baseUrl` should be `null` for models that aren't served from the repo in demo deployments
- Never publish from `models/local/` — always from `/media/x/models/rdrr/<model-id>/`
