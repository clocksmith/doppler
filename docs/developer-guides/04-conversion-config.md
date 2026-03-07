# Add a Conversion Config

## Goal

Add a reproducible conversion config for a checkpoint that already fits an existing preset and runtime path.

## When To Use This Guide

- You want a checked-in conversion recipe under `tools/configs/conversion/`.
- The model family, kernel path, and quantization workflow already exist.

## Blast Radius

- JSON only

## Required Touch Points

- `tools/configs/conversion/<family>/<model>.json`
- Optional `tools/configs/conversion/README.md` note if the config becomes a maintained example

## Recommended Order

1. Copy the closest existing config in `tools/configs/conversion/`.
2. Set `output.baseDir` and `output.modelBaseId`.
3. Set `presets.model`, `quantization`, and `inference.defaultKernelPath`.
4. Add `inference.sessionDefaults` or `inference.execution` only when the manifest should pin explicit execution-v0 state.
5. Run conversion, then run a real verify or debug pass against the produced artifact.

## Verification

- `npm run onboarding:check`
- Run `convert` with the checked-in config
- Run `verify` or `debug` on the produced artifact
- Review deterministic output quality before any promotion or publication step

For command-shape examples, use [../getting-started.md](../getting-started.md) and `tools/configs/conversion/README.md`.

## Common Misses

- Running conversion ad hoc and not checking the config into `tools/configs/conversion/`.
- Stopping at manifest or load validation without a coherence check.
- Forgetting that `output.modelBaseId` is authoritative for the emitted model ID.
- Assuming the browser OPFS store is populated automatically by the convert step.
- Forgetting that `inference.defaultKernelPath` can auto-generate execution-v0 data when no explicit `inference.execution` is present.

## Related Guides

- [03-model-preset.md](03-model-preset.md)
- [05-promote-model-artifact.md](05-promote-model-artifact.md)
- [06-kernel-path-preset.md](06-kernel-path-preset.md)

## Canonical References

- `tools/configs/conversion/README.md`
- [../conversion-runtime-contract.md](../conversion-runtime-contract.md)
- [../getting-started.md](../getting-started.md)
- `src/converter/conversion-plan.js`
