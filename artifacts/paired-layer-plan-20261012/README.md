# Paired layer orchestration controls

Component: `doppler.runtime-source.inference.pipelines.text`
Intent: preserved. Execute the approved paired intermediate chain without
reinterpreting two storage components as separate logical activation values.

The declared layer executor retains representation through each activation
Tensor, norm weight binding, and output probe. It rejects unsupported scalar
conversions and validates returned shape/representation. A rejected kernel
output is owned before dtype validation, so the failure path releases it.
Norm weight upload follows input validation and has deterministic cleanup.
Saving/loading the same buffer into an existing slot no longer increments a
reference without creating another owner.

Mac Metal/Dawn physical controls:

- `tests/inference/paired-ffn-physical.test.js`: actual gate/up projection,
  SiLU multiplication and down projection, two tokens with hidden size 128 and
  intermediate size 64. Immediate and recorded maximum absolute error
  `1.491862189340054e-16` against independent F64 equations; tolerance `1e-11`.
- `tests/inference/layer-plan-paired-physical.test.js`: normalization, saved
  aliases, residual addition and cleanup. Immediate and recorded maximum
  absolute error `7.993605777301127e-15`; tolerance `1e-11`. Output-dtype
  rejection releases intermediates; unsupported casts reject before execution.
- Existing scalar LayerNorm and LFM2 convolution controls pass. Their scripts
  were imported through Node wrappers that call `destroyDevice()` afterward.

The accompanying text receipts retain exact operands, independent expectations,
physical outputs and provider identity. Source identities cover the scoped
integration and controls. `layer-plan-paired-contract.test.js` additionally
uses synthetic dispatch to inject lost metadata and normalization failures,
checking immediate/recorded ownership and rejection before norm uploads.

Boundary effects: consumes paired Tensor/kernel contracts and explicit
`LayerContext` representation supplied by the selected execution plan. No model
artifact, approved F64 reference, F16 cache format or application pin changes.

These small controls do not qualify the complete model, either distributed
placement, AMD layer execution, signed opening, simultaneous long conversations,
the allocation ceiling, or document-answer quality. Those gates remain open.

Source-style validation passes. Concurrent full-source type/architecture checks
currently report in-progress attention dispatch typing and an attention executor
line ceiling owned by the parallel attention task. They are not reported as
passing checks for this working tree; the full package sweep remains pending.
