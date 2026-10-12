# Approved paired-intermediate implementation: operator checkpoint

The user approved the broader paired-intermediate chain after the retained memory review. This checkpoint implements explicit interleaved high/low F32 operators; it does **not** qualify a complete model. The approved independent F64 target, 0.001 tolerance, model weights, F16 KV storage, and 1,420,000,000-byte executor ceiling remain unchanged. Full graph binding, whole-model accuracy, concurrent memory, and useful-answer qualification remain open. Model execution requesting the incomplete paired graph fails closed.

## Physical operator evidence

Mac Metal and AMD Vulkan independently passed the same bounded controls:

- Elementary operations: 20,570 outputs, normalized tolerance 2e-12. Maximum error 4.5474518248211915e-13 (sine). The initial range-reduction failure and an AMD receipt-write failure remain retained separately.
- RMSNorm: six width/phase/residual/weight-offset cases, source epsilon preserved as high/low uniforms. Maximum normalized error 5.808523622347618e-15.
- Projection: six F32/F16/Q4K-weight and paired/scalar-output cases. Paired maximum normalized error 2.9976021664879227e-14; scalar public outputs match independently rounded F64 result bits.
- Direct paired-to-F16 storage uses exact integer midpoint decisions. The earlier Number-reference receipts remain historical; the later exact BigInt oracle and physical public cast controls are retained in the paired-activation contract evidence. No Number(high)+Number(low) oracle is used to decide tiny-residual half ties.

Each JSON receipt binds tested source, input fixture, platform, and outcome. These are correctness controls, not performance measurements. Public idle contributors may retain unrelated GPU resources; no foreground budget or speed claim follows.

## Contracts and boundaries

A paired Tensor explicitly declares precision `two-f32`, components `2`, and elementStrideBytes `8`. Weights retain their original representation. No JavaScript tensor arithmetic or implicit widening is introduced. Final public logits can explicitly request scalar F32. Direct paired-to-half conversion avoids an intervening F32 rounding. Finite operands and finite intermediate exponent range remain required; this representation does not extend the F32 exponent range.

Immediate and recorded wrapper tests cover borrowed/owned output, malformed metadata, geometry, alias rejection, allocation failure, and failed dispatch cleanup. The generated arithmetic comes from canonical WGSL include files; the unreferenced historical duplicate was moved byte-for-byte into `historical-sources` with provenance.

Component: `doppler.gpu.kernels` and text execution configuration. Intent: preserved under the user's explicit precision extension. Boundary effects: explicit paired Tensor/partition ABI and execution-plan admission; accepted application archive unchanged.
