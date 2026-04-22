# Add a Conversion Config

## Goal

Add a reproducible conversion config for a checkpoint that already fits an existing runtime path.

## When To Use This Guide

- You want a checked-in conversion recipe under `src/config/conversion/`.
- The model family, execution graph pattern, and quantization workflow already exist.

## Blast Radius

- JSON only

## Required Touch Points

- `src/config/conversion/<family>/<model>.json`
- Optional `src/config/conversion/README.md` note if the config becomes a maintained example

## Recommended Order

1. Copy the closest existing config in `src/config/conversion/`.
2. Set `output.baseDir` and `output.modelBaseId`.
3. Set `quantization` and any manifest-owned inference fields needed by the artifact.
4. Add the explicit execution graph (kernels, decode, prefill, preLayer, postLayer, and policies).
5. Run conversion, then run a real verify or debug pass against the produced artifact.

## Verification

- `npm run onboarding:check`
- Run `convert` with the checked-in config
- Run `verify` or `debug` on the produced artifact
- Review deterministic output quality before any promotion or publication step

For command-shape examples, use [../getting-started.md](../getting-started.md) and `src/config/conversion/README.md`.

## Common Misses

- Running conversion ad hoc and not checking the config into `src/config/conversion/`.
- Stopping at manifest or load validation without a coherence check.
- Forgetting that `output.modelBaseId` is authoritative for the emitted model ID.
- Assuming the browser OPFS store is populated automatically by the convert step.
- Omitting the `inference.execution` graph — the converter only accepts v1 configs with an explicit execution graph.

## Related Guides

- [05-promote-model-artifact.md](05-promote-model-artifact.md)
- [06-kernel-path-config.md](06-kernel-path-config.md)

## Canonical References

- `src/config/conversion/README.md`
- [../conversion-runtime-contract.md](../conversion-runtime-contract.md)
- [../getting-started.md](../getting-started.md)
- `src/converter/conversion-plan.js`
