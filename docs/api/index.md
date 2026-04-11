# API Docs

Canonical public API documentation for `doppler-gpu`.

Use this section for package surfaces that are intentionally exported and meant to be consumed directly.
The generated export inventory lives under [reference/exports.md](reference/exports.md).

## Surface Classes

### Primary app-facing surface

- [Root API](root.md) - `doppler-gpu`

### Advanced public surfaces

- [Advanced Export Map](advanced-root-exports.md) - migration map from the old broad root surface to dedicated advanced subpaths
- [Loaders API](loaders.md) - `doppler-gpu/loaders`
- [Orchestration API](orchestration.md) - `doppler-gpu/orchestration`
- [Generation API](generation.md) - `doppler-gpu/generation`
- [Diffusion API](diffusion.md) - `doppler-gpu/diffusion`
- [Energy API](energy.md) - `doppler-gpu/energy`
- [Tooling API](tooling.md) - `doppler-gpu/tooling`, including the shared `convert|debug|bench|verify|diagnose|lora|distill` command contract

## Documentation Form

Each manual API page follows the same structure:

1. Purpose
2. Import path
3. Audience
4. Stability
5. Primary exports
6. Minimal example
7. Code pointers
8. Related surfaces

Generated reference pages provide:

- export inventory by subpath
- source `.d.ts` and `.js` pointers
- machine-derived symbol lists from shipped declaration files

## Source Of Truth

- package export map: [package.json](../../package.json)
- shipped type surfaces under `src/**/*.d.ts`
- manual behavior docs in this directory

## Current Classification

### Stable and preferred

- `doppler-gpu`

### Public but advanced

- `doppler-gpu/loaders`
- `doppler-gpu/orchestration`
- `doppler-gpu/generation`
- `doppler-gpu/diffusion`
- `doppler-gpu/energy`
- `doppler-gpu/tooling`
