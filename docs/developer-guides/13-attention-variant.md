# Add an Attention Variant

## Goal

Add a new attention mechanism within the existing transformer text pipeline.

## When To Use This Guide

- The model still belongs to the transformer text pipeline.
- Attention behavior changes, but you do not need a brand-new pipeline family.

## Blast Radius

- Deep pipeline + kernels + tests

## Required Touch Points

- `src/inference/pipelines/text/attention/run.js`
- `src/inference/pipelines/text/attention/record.js`
- `src/inference/pipelines/text/attention/projections.js` when QKV setup changes
- `src/gpu/kernels/attention*.wgsl` and `src/gpu/kernels/attention.js`
- `src/inference/kv-cache/*` or `src/inference/kv-cache.js` if layout interactions change
- conversion config `inference.execution` and execution graph transforms
- Optional new manifest/runtime fields and conversion-config updates
- Kernel tests and end-to-end pipeline tests

## Recommended Order

1. If the variant needs new config fields, finish [07-manifest-runtime-field.md](07-manifest-runtime-field.md) first.
2. Implement decode and prefill kernels or explicitly fail closed for unsupported phases.
3. Wire the variant through the attention run and record paths.
4. Update KV-cache integration if read or write behavior changes.
5. Add or update conversion execution graphs and any needed graph transforms.
6. Add segment tests and one end-to-end debug comparison against a trusted reference.

## Verification

- `npm run test:gpu`
- `npm run test:gpu:browser`
- Run an end-to-end `debug` pass and compare output behavior against the reference stack you use for that model family

## Common Misses

- Implementing only decode or only prefill for a variant that actually needs both.
- Forgetting browser verification. Attention path bugs commonly diverge across Node and browser WebGPU.
- Ignoring KV-cache compatibility and silently reading the wrong layout.
- Hiding variant selection in JS conditionals instead of config and rule assets.

## Related Guides

- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [06-kernel-path-config.md](06-kernel-path-config.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [15-kvcache-layout.md](15-kvcache-layout.md)
- [composite-model-family.md](composite-model-family.md)

## Canonical References

- `src/inference/pipelines/text/attention/run.js`
- `src/inference/pipelines/text/attention/record.js`
- `src/gpu/kernels/attention.js`
- `src/inference/README.md`
- [../kernel-testing-design.md](../kernel-testing-design.md)
