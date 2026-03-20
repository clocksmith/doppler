# Add an Activation Implementation

## Goal

Add a new FFN activation implementation across WGSL, runtime wiring, and config.

## When To Use This Guide

- A model family needs an activation Doppler does not currently implement.
- Existing FFN rule maps and kernels cannot express the new behavior.

## Blast Radius

- WGSL + runtime + config + tests

## Required Touch Points

- `src/gpu/kernels/<activation>.wgsl`
- `src/gpu/kernels/<activation>.js` and `.d.ts`
- `src/gpu/kernels/index.js` and `.d.ts`
- `src/rules/inference/ffn.rules.json` if FFN activation selection needs a new rule outcome
- `src/inference/pipelines/text/ffn/dense.js`
- If manifest values need a new enum:
  `src/config/schema/inference.schema.d.ts`
  and
  `src/config/schema/manifest.schema.d.ts`
- Conversion logic or config assets that should emit the new activation value
- Kernel and runtime tests

## Recommended Order

1. Implement the WGSL kernel and JS wrapper with matching run and record paths.
2. Export the wrapper through `src/gpu/kernels/index.js` and `.d.ts`.
3. Update `src/rules/inference/ffn.rules.json` or the FFN call site so the new activation can be selected explicitly.
4. Add schema and conversion-config changes if the activation name must be authored into manifests.
5. Add tests for the kernel and at least one FFN path that uses it.

## Verification

- `npm run kernels:check`
- `npm run test:gpu`
- Run one debug pass with a model or test fixture that selects the new activation

## Common Misses

- Adding the kernel but not updating the FFN rule map, leaving the implementation unreachable.
- Exporting the run path but not the record path.
- Updating the runtime call site without updating declaration files.
- Adding a new manifest value without updating the relevant schema types.

## Related Guides

- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [13-attention-variant.md](13-attention-variant.md)
- [composite-model-family.md](composite-model-family.md)

## Canonical References

- `src/gpu/kernels/gelu.js`
- `src/gpu/kernels/silu.js`
- `src/rules/inference/ffn.rules.json`
- `src/inference/pipelines/text/ffn/dense.js`
- [../style/wgsl-style-guide.md](../style/wgsl-style-guide.md)
