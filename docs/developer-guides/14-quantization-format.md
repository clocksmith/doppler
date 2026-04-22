# Add a Quantization Format

## Goal

Add a new weight quantization format from conversion through runtime execution.

## When To Use This Guide

- Existing manifest quantization values and loader paths cannot represent the new format.
- The work spans converter output, manifest metadata, loader logic, and runtime kernels.

## Blast Radius

- Converter + loader + kernels + tests

## Required Touch Points

- `src/config/schema/converter.schema.js`
- `src/config/schema/manifest.schema.js` and `.d.ts`
- `src/converter/quantization-info.js`, `src/converter/conversion-plan.js`, and quantizer code under `src/converter/`
- Loader code under `src/loader/`
- Dequant or fused runtime kernels under `src/gpu/kernels/`
- Kernel-path configs and any conversion configs that should use the new format
- Converter, kernel, and end-to-end tests

## Recommended Order

1. Define the manifest and converter vocabulary for the new format.
2. Implement converter-side quantization info and emitted bytes.
3. Implement loader-side interpretation and runtime kernel support.
4. Add or update conversion execution graphs and configs.
5. Add reference tests for conversion and dequant behavior.
6. Run one real convert plus debug or verify pass with a model that uses the new format.

## Verification

- `npm run test:unit`
- `npm run test:gpu`
- Run one convert plus debug or verify flow and confirm the converted model produces coherent output

## Common Misses

- Adding the manifest string without teaching the loader and runtime how to consume it.
- Emitting incomplete `quantizationInfo` metadata.
- Reusing an existing execution graph whose assumptions do not match the new format.
- Verifying byte layout only, without checking actual model output.
- Forgetting browser verification when the runtime path depends on WebGPU kernels.

## Related Guides

- [04-conversion-config.md](04-conversion-config.md)
- [06-kernel-path-config.md](06-kernel-path-config.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [composite-model-family.md](composite-model-family.md)

## Canonical References

- [../conversion-runtime-contract.md](../conversion-runtime-contract.md)
- `src/config/schema/converter.schema.js`
- `src/config/schema/manifest.schema.d.ts`
- `src/converter/quantization-info.js`
- `src/loader/index.js`
- `tests/converter/quantizer.test.js`
