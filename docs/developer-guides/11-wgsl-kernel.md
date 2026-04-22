# Add a WGSL Kernel or Kernel Variant

## Goal

Add a new GPU kernel implementation or a new variant of an existing kernel.

## When To Use This Guide

- Existing kernels and entry points cannot express the needed algorithm or hardware path.
- The work belongs in the compute plane, not just in config or rule maps.

## Blast Radius

- WGSL + wrapper/selection + tests

## Required Touch Points

- `src/gpu/kernels/<name>.wgsl`
- `src/gpu/kernels/<name>.js` and `.d.ts`
- `src/gpu/kernels/index.js` and `.d.ts`
- Optional selection logic in `src/rules/**/*.json`
- Optional conversion execution graph or graph-transform updates when the new kernel changes execution identity
- Kernel tests and browser harness coverage

## Recommended Order

1. Use the WGSL topology test first: decide whether you need a new file, a new entry point, or only new `override` constants.
2. Implement the WGSL kernel.
3. Add the JS wrapper and matching declaration file under `src/gpu/kernels/`.
4. Re-export it through `src/gpu/kernels/index.js` and `.d.ts`.
5. Update rule maps, conversion execution graphs, or graph transforms only if the new kernel must be selected or named as a new execution identity.
6. Add correctness tests and browser harness coverage.
7. Run kernel generation or checks if the kernel participates in generated variants.

## Verification

- `npm run kernels:check`
- `npm run test:gpu`
- `npm run test:gpu:browser`
- Run one debug or verify flow that exercises the new kernel in a real path

## Common Misses

- Treating `src/gpu/kernel-selector.js` as the main implementation point. New work belongs under `src/gpu/kernels/`.
- Forgetting `.d.ts` and index exports.
- Changing execution identity when the kernel is only a selection detail, or skipping the execution graph update when identity really changed.
- Testing only in Node and missing browser-only shader or pipeline issues.
- Forgetting `npm run kernels:generate` or `npm run kernels:check` when generated variants are involved.

## Related Guides

- [06-kernel-path-config.md](06-kernel-path-config.md)
- [10-activation-implementation.md](10-activation-implementation.md)
- [13-attention-variant.md](13-attention-variant.md)
- [14-quantization-format.md](14-quantization-format.md)

## Canonical References

- `src/gpu/kernels/index.js`
- `src/gpu/kernels/gelu.js`
- `src/gpu/kernels/attention.js`
- [../style/javascript-style-guide.md](../style/javascript-style-guide.md)
- [../style/wgsl-style-guide.md](../style/wgsl-style-guide.md)
- [../kernel-testing-design.md](../kernel-testing-design.md)
