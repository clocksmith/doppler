# DOPPLER Config System

Implementation notes for `src/config/*`.

Runtime-facing config behavior is canonical in
[../../docs/config.md](../../docs/config.md).

## Scope

This directory owns:
- schema/default definitions
- config loading and merge helpers
- execution graph transforms for capability adaptation
- config validation utilities

## Directory map

```text
src/config/
├── runtime.js                   # runtime get/set and resolved config access
├── merge.js                     # canonical merge behavior
├── kernel-path-loader.js        # kernel-path utility functions for inline paths
├── conversion/                  # conversion configs
├── diagnostics/                 # non-runtime diagnostic policy assets
├── runtime/                     # runtime profiles only
├── support-tiers/               # canonical subsystem support registry
├── schema/                      # schemas + defaults
├── transforms/                  # execution graph transforms for capability adaptation
└── platforms/                   # platform config assets
```

## Maintainer rules

- Keep runtime behavior config-first; avoid hardcoded fallbacks in runtime paths.
- Capability adaptation uses composable transforms on the manifest execution graph.
- Keep merge semantics centralized in config utilities (no ad-hoc deep merges in runners).
- Preserve `null` vs `undefined` semantics required by schema contracts.

## Internal workflows

Validate config integrity:

```bash
npm run kernels:check
npm run config:single-source:check
npm run support:matrix:check
npm run support:subsystems:check
npm run onboarding:check:strict
```

## When adding a model family

1. Add or update a conversion config in `src/config/conversion/`.
2. Verify the execution graph works with existing transforms (`removeSubgroups`, `widenToF32Activations`).
3. Add tests for merge/selection behavior and run validation commands.

## Related

- [../../docs/config.md](../../docs/config.md)
- [../../docs/style/config-style-guide.md](../../docs/style/config-style-guide.md)
- [../../docs/conversion-runtime-contract.md](../../docs/conversion-runtime-contract.md)
