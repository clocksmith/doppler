# Add a Kernel-Path Config

## Goal

Define a new execution plan using existing kernels.

## When To Use This Guide

- You want a different decode or prefill path without writing new WGSL.
- The change is about execution identity, dtype mix, or kernel sequencing.

## Blast Radius

- JSON + registry

## Required Touch Points

- `src/config/kernel-paths/<id>.json`
- `src/config/kernel-paths/registry.json`
- Usually one of:
  `src/config/conversion/<family>/<model>.json`

## Recommended Order

1. Copy the nearest existing kernel-path config.
2. Update `activationDtype`, `kvDtype`, and the relevant `decode`, `prefill`, `preLayer`, `postLayer`, or `sampling` steps.
3. Add the new ID to `src/config/kernel-paths/registry.json`.
4. Wire the new ID into a conversion config or runtime profile so it is reachable.
5. Run kernel generation checks and one debug run with the path explicitly selected.

## Verification

- `npm run onboarding:check`
- `npm run kernels:check`
- Run one debug pass with the new path forced through runtime config

## Common Misses

- Adding the config file but not the registry entry.
- Registering the config but never wiring it into a conversion config or runtime profile.
- Referencing a kernel filename or entry point that does not exist.
- Making decode and prefill inconsistent when the model needs both paths.
- Using semantic alias IDs instead of explicit path IDs that encode the real execution behavior.

## Related Guides

- [03-model-family-config.md](03-model-family-config.md)
- [04-conversion-config.md](04-conversion-config.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [13-attention-variant.md](13-attention-variant.md)

## Canonical References

- `src/config/kernel-paths/registry.json`
- `src/config/kernel-paths/gemma3-q4k-dequant-f32a-online.json`
- [../config.md](../config.md)
- `src/inference/pipelines/text/model-load.js`
