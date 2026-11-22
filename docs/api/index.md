# API Docs

Canonical public API documentation for `dopplerx`.

Use this section for package surfaces that are intentionally exported and meant to be consumed directly.
The generated export inventory lives under [reference/exports.md](reference/exports.md).

## Surface Classes

### Primary app-facing surface

- [Doppler Run API](root.md) - start with `dopplerx/host`; use `dopplerx`
  when supplying explicit execution ports
- [Compatibility API](compat.md) - `dopplerx/compat`
- [Choice scoring](choice-scoring.md) - typed, contextual answer scores and operation qualification
- [Target sequence scoring](target-sequence-scoring.md) - teacher-forced log-probability of an exact target continuation

### Exported advanced surfaces

- [Advanced Export Map](advanced-root-exports.md) - migration map from the old broad root surface to dedicated advanced subpaths
- [Generation API](generation.md) - `dopplerx/generation`
- [Tooling API](tooling.md) - `dopplerx/tooling`, with a tier1 command/storage core plus experimental operator flows

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

Deprecated subpaths are listed with their removal release in the
[changelog](../../CHANGELOG.md) and the [export inventory](reference/exports.md).

### Stable and preferred

- `dopplerx/host` composes existing host ports; model/host qualification is separate
- `dopplerx`
- `dopplerx/runtime` (compatibility alias of `dopplerx`; `createDopplerRuntime` aliases `createDopplerRun`)

### Tier1 advanced

- `dopplerx/generation`
- tier1 portions of `dopplerx/tooling`

### Exported but experimental

- `dopplerx/compat`
- experimental operator portions of `dopplerx/tooling`
