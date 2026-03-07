# API Docs

Canonical public API documentation for `@simulatte/doppler`.

Use this section for package surfaces that are intentionally exported and meant to be consumed directly.
The generated export inventory lives under [reference/exports.md](reference/exports.md).

## Surface Classes

### Primary app-facing surface

- [Root API](root.md) - `@simulatte/doppler`

### Advanced public surfaces

- [Advanced Root Exports](advanced-root-exports.md) - root-level loaders, pipelines, adapters, and compatibility re-exports
- [Provider API](provider.md) - `@simulatte/doppler/provider`
- [Generation API](generation.md) - `@simulatte/doppler/generation`
- [Diffusion API](diffusion.md) - `@simulatte/doppler/diffusion`
- [Energy API](energy.md) - `@simulatte/doppler/energy`
- [Tooling API](tooling.md) - `@simulatte/doppler/tooling`, including the shared `convert|debug|bench|verify|lora|distill` command contract

### Exposed but not primary doc targets

- `@simulatte/doppler/internal`

This surface is reachable through the package export map, but it is not the primary stability story for application developers.
Use it only when you deliberately want lower-level or internal source access.

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

- `@simulatte/doppler`

### Public but advanced

- root-level advanced exports documented in [advanced-root-exports.md](advanced-root-exports.md)
- `@simulatte/doppler/provider`
- `@simulatte/doppler/generation`
- `@simulatte/doppler/diffusion`
- `@simulatte/doppler/energy`
- `@simulatte/doppler/tooling`

### Publicly reachable but not recommended as the main contract

- `@simulatte/doppler/internal`
