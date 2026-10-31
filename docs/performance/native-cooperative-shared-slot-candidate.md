# Native cooperative and shared-slot candidates

These opt-in engine programs adapt the cooperative quarter-block GEMV and
shared-slot GU/SiLU/down/combine topology inspected in the on-host Pooled
snapshot verified against commit `6b4a24f73cce7ef9a9e3ab35b9a0d4c09b51dfff`. That snapshot has no independent `.git` directory;
its enclosing repository HEAD is not Pooled's revision. Inspected source SHA-256:

- `engine/wgsl/coop.js`: `2c552bb7849a272808f8c5779288a3f9b954fab0d0bb65f7a88d4c468bbc6963`
- `engine/wgsl/moe.js`: `37b2cf395f540f5b3f24aed3c79af1bd1b91fc292e9e58207d019837f77e5ce9`
- `LICENSE`: `5e78649f9b4f053fe724537afadaadfb38f3cd9dcd7f113bf74feadda3ca82f2`

Derived WGSL and its code-generation patches carry the full MIT copyright and
permission notice. Pooled's split quant/scale buffers are adapted to unchanged
interleaved GGUF blocks. Q4_0 low/high nibbles and Q8_0 signed bytes retain their
actual scales. Shared-slot execution also admits native Q5_0 and Q4_1 through
the existing block decoder; no tensor is silently re-quantized.

`matmul/q4_0_cooperative` and `matmul/q8_0_cooperative` reuse activation quarters
across four rows and reduce F32 partials. Existing scalar GEMV and tiled prefill
remain available. Registry geometry and the existing Q4 wrapper validate the
program, layout, buffer extents, and device bounds.

The shared-slot programs require both `moe_shared_slot_gu` and
`moe_shared_slot_down` in the effective phase/layer path. Gate/up and SiLU run for
routed slots plus one shared slot. Down projects and combines those slots in
route order. The shared scalar router remains a separate projection. Doppler's
refined, clamped sigmoid/SiLU and product clamp are retained. The output contains
no outer residual; the existing layer adds it exactly once. Prepared decode
workspace buffers and recorder cleanup keep their existing ownership.

The existing JSON registry, WGSL patch generator, kernel wrapper, Qwen route
executor, and layer-path resolver own these contracts. No default kernel choice,
model bytes, or model promotion is changed. The internal
`tools/experiments/native-decode-candidate.js` accepts an inline path, layer
count, explicit projection variants, and shared-slot layer indices. It expands
effective per-layer paths before editing decode, preserving each prefill path,
weight reference, and untouched operation. Full-model selection still requires
actual materialized weights to pass runtime admission.

The native artifact retains 70 tensors Pooled re-quantizes to Q8: 60 shared Q5_0,
five expert-down Q4_1, four attention-Q Q6_K, and one output-head Q6_K. This native
lane is not equal weights to Pooled. Equal-materialization and best-native
performance remain separate comparisons. F32 activations are independent of
weight/KV storage. All 40 effective layers must remain present.

CPU tests cover admission, per-layer preservation, explicit phase forwarding,
and cleanup after GU/down failure. Separate software-only tests compile the
programs and exercise bounded reference fixtures; they do not qualify Radeon.
The physical numerical test rejects software adapters. Full-model numerics,
continuation, and matched performance receipts are still required; dispatch
count and source topology establish no speed claim. See `JUPITER_STATUS.md` for
this integration's checks, environment blockers, and lock-invocation deviation.
