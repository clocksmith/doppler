# Add an Attention Variant

## Goal

Add a new attention mechanism within the existing transformer text pipeline.

## When To Use This Guide

- The model still belongs to the transformer text pipeline.
- Attention behavior changes, but you do not need a brand-new pipeline family.

## Blast Radius

- Deep pipeline + kernels + tests

## Required Touch Points

- `src/inference/pipelines/text/attention/plan.js` for semantic decisions
- `src/inference/pipelines/text/attention/interpreter.js` for the one concrete operation path
- `src/inference/pipelines/text/attention/executor-immediate.js` and
  `executor-recorded.js` only as submission adapters
- `src/inference/pipelines/text/attention/projections.js` when QKV setup changes
- `src/gpu/kernels/attention*.wgsl` and `src/gpu/kernels/attention.js`
- `src/inference/kv-cache/*` if layout interactions change
- conversion config `inference.execution` and execution graph transforms
- Optional new manifest/runtime fields and conversion-config updates
- Kernel tests and end-to-end pipeline tests

## Recommended Order

1. If the variant needs new config fields, finish [07-manifest-runtime-field.md](07-manifest-runtime-field.md) first.
2. Implement decode and prefill kernels or explicitly fail closed for unsupported phases.
3. Add the variant decision to the immutable semantic attention plan.
4. Update KV-cache integration if read or write behavior changes.
5. Bind and execute it once in the interpreter; keep both submission adapters decision-free.
6. Add or update conversion execution graphs and any needed graph transforms.
7. Add semantic-plan parity, segment, resource-lifecycle, and end-to-end debug tests.

## Verification

- `npm run test:gpu`
- `npm run test:gpu:browser`
- Run an end-to-end `debug` pass and compare output behavior against the reference stack you use for that model family

## Common Misses

- Implementing only decode or only prefill for a variant that actually needs both.
- Forgetting browser verification. Attention path bugs commonly diverge across Node and browser WebGPU.
- Ignoring KV-cache compatibility and silently reading the wrong layout.
- Hiding variant selection in JS conditionals instead of config and rule assets.
- Duplicating a dtype, kernel, fusion, layout, gate, geometry, or output decision
  between the immediate and recorded executors.
- Putting GPU buffers, recorder objects, closures, or cleanup callbacks in the
  semantic plan.

## Recurrent source-layout admission

The GGUF retained-state prefill candidate uses the existing
`useLinearAttentionRegisterPrefill` opt-in and the registered
`gated_delta_recurrent/gguf_register_prefill` variant. It admits scalar F32
input/output/state, one state component, SiLU norm gating, K=128, V<=128 and
explicit consistent head geometry. Single-token tails retain the GGUF baseline;
GGUF register/fused decode and two-component state are not admitted.

Preserve the source layout's arithmetic and head mapping when adding a variant:
GGUF uses `head % NK` and epsilon inside the Q/K L2 root. HF register kernels
use a different head mapping. Retain nonzero initial state, chunk/decode
continuation, final state, packed-input addressing and norm/gate order in both
submission adapters. A session flag cannot authorize a kernel outside the
prepared program's pinned closure.

Operator fixtures and a faster recurrent pass do not qualify a model or a
platform. Require full-model numerical/quality and phase-specific performance
receipts before selecting this candidate in a production artifact/profile.

## Related Guides

- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [06-kernel-path-config.md](06-kernel-path-config.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [15-kvcache-layout.md](15-kvcache-layout.md)
- [composite-model-family.md](composite-model-family.md)

## Canonical References

- `src/inference/pipelines/text/attention/plan.js`
- `src/inference/pipelines/text/attention/interpreter.js`
- `src/inference/pipelines/text/attention/executor-immediate.js`
- `src/inference/pipelines/text/attention/executor-recorded.js`
- `src/inference/resource-scope.js`
- `src/gpu/kernels/attention.js`
- `src/inference/GUIDE.md`
- [../kernel-testing-design.md](../kernel-testing-design.md)
