# DOPPLER Config System

Implementation notes for `src/config/*`.

Runtime-facing config behavior is canonical in
[../../docs/config.md](../../docs/config.md).

## Scope

This directory owns:
- schema/default definitions
- config loading and merge helpers
- kernel-path registry loading
- config validation utilities

## Directory map

```text
src/config/
├── runtime.js                   # runtime get/set and resolved config access
├── merge.js                     # canonical merge behavior
├── kernel-path-loader.js        # kernel-path registry/path resolution
├── conversion/                  # conversion configs
├── runtime/                     # runtime profiles and overlays
├── schema/                      # schemas + defaults
├── kernel-paths/                # kernel-path registries
└── platforms/                   # platform config assets
```

## Maintainer rules

- Keep runtime behavior config-first; avoid hardcoded fallbacks in runtime paths.
- Treat `registry.json` as source-of-truth for kernel-path identity/lifecycle.
- Keep merge semantics centralized in config utilities (no ad-hoc deep merges in runners).
- Preserve `null` vs `undefined` semantics required by schema contracts.

## Internal workflows

Validate config integrity:

```bash
npm run kernels:check
npm run support:matrix:check
npm run onboarding:check:strict
```

Inspect kernel-path registry IDs:

```bash
node -e "const fs=require('node:fs');const r=JSON.parse(fs.readFileSync('src/config/kernel-paths/registry.json','utf8'));console.log(r.entries.map((e)=>e.id).join('\n'));"
```

Inspect legacy aliases:

```bash
node -e "const fs=require('node:fs');const r=JSON.parse(fs.readFileSync('src/config/kernel-paths/registry.json','utf8'));console.log(r.entries.filter((e)=>e.status==='legacy').map((e)=>`${e.id} -> ${e.aliasOf}`).join('\n'));"
```

## When adding a model family

1. Add or update a conversion config in `src/config/conversion/`.
2. Add or adjust kernel-path configs in `src/config/kernel-paths/`.
3. Register IDs/status in `src/config/kernel-paths/registry.json`.
4. Add tests for merge/selection behavior and run validation commands.

## Related

- [../../docs/config.md](../../docs/config.md)
- [../../docs/style/config-style-guide.md](../../docs/style/config-style-guide.md)
- [../../docs/conversion-runtime-contract.md](../../docs/conversion-runtime-contract.md)
