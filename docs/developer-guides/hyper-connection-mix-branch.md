# Opt-in refined HC mix and branch

`hc_mix_branch` replaces the separate HC sigmoid and branch-reduction dispatches
with one registered `hyper_connection_mix_branch/f32` operator. No checked-in
model or runtime profile enables it. It does not change weights, representation,
norms, low-rank projections, injection, attention, routing or expert arithmetic.

Add this step to the selected phase's actual layer steps (including each active
layer override). The ordinary kernel-path merge and declaration types apply:

```json
{
  "op": "hc_mix_branch",
  "kernel": "hyper_connection_mix_branch.wgsl",
  "entry": "main",
  "digest": "sha256:c51798006b3d886c90f24cb76f70e6779f3a731cf23360571171cf18415382ab"
}
```

Absent declarations retain the separate operators. Duplicate declarations,
unregistered shaders/entries, missing or incorrect compiled entry digests, and
overrides of model geometry fail closed.
Selection uses the current phase and layer; the root mixer uses layer zero's
phase declaration, as its projections do. A decode-only declaration leaves
prefill unchanged. This adds no session flag or model-default policy.

The shader copies the existing refined exponential, reciprocal, division and
sigmoid functions. For every stream, sigmoid outputs pass through workgroup
memory to a different invocation before multiplication. Products pass through
a second exchange before addition. The sum follows the existing stream order
and refined final division. These two observable F32 materialization boundaries
prevent contracting the product into accumulation or inlining sigmoid arithmetic
through its former output-store boundary. Tail lanes participate in all three
barriers and never read beyond the input.

For recorded single-token calls with an existing prepared decode workspace,
only the fused branch output is borrowed from `hc:<layer>:<role>:branch`.
Attention, MLP and root roles have separate slots; other temporaries retain
existing ownership. Queue order protects reuse by serial decode calls. The
session owns the slot until all submitted work settles and the workspace is
released. Immediate and multi-token calls use ordinary pooled outputs.

CPU checks:

```sh
node tools/run-node-tests.js tests/kernels/hyper-connection-mix-branch-contract.test.js tests/inference/hyper-connection-mix-branch-wiring.test.js tests/inference/hyper-connection-layer-wiring.test.js tests/inference/hyper-connection-sequence.test.js
```

After acquiring a clean GPU window, run the short synthetic component test:

```sh
node tests/kernels/hyper-connection-mix-branch-gpu.test.js
```

It rejects software adapters and checks exact output bits against the actual
separate kernels, in immediate, recorded and repeated prepared-output modes,
with irregular widths/stream counts and saturated logits. All observed GPUBuffer
allocations have an 8 MiB cap; explicit retirement must reach zero before device
destruction can clear allocation accounting. This is a component
parity check, not model correctness, complete-request performance or physical
multi-model retirement qualification. Freeze a new source/profile identity and
rerun the required exact-token and complete-request gates before any promotion.

## Q4_0 up projection candidate

A separate opt-in single-token program also absorbs the up projection for the
attention and MLP HC sites. Declare each intended role independently in its
actual decode layer steps, alongside its existing `attn_hc_up` or `mlp_hc_up`
step. For example:

```json
{
  "op": "attn_hc_up_mix_branch",
  "kernel": "hyper_connection_up_mix_branch_q4_0.wgsl",
  "entry": "main",
  "digest": "sha256:8d5cfc4c6ac02fe00f8f1dd97feb07de7e713c4bf343cf0eabf46afa4b26273b"
}
```

Use `mlp_hc_up_mix_branch` for the other site. The original up role must resolve
to `matmul/q4_0_gemv` (`matmul_q4_0.wgsl`, entry `gemv`, workgroup 128,
four rows, 32 lanes per row). Cooperative, pair, other dtype programs and
prefill declarations are rejected. Original norm/down/SiLU and injection
remain unchanged. The root mixer has BF16 source projections materialized for
its declared F32 program and stays on that independently admitted path; it is
never silently routed through this Q4_0 specialization.

The fused program groups four rows from the same output column across the
four residual streams. Each row retains the original Q4_0 byte decoding,
32-lane strided FMA sequence and shared reduction tree (16, 8, 4, 2, 1).
Projection outputs, refined sigmoid results and products each cross an
invocation boundary through workgroup memory before their next consumer.
Products sum in stream order and use the same refined division. No projection
precision or representation changes. It removes the up-projection output and
one dispatch compared with the prior fused-mix candidate. Performance remains
unproved until a matched complete request passes.

CPU checks:

```sh
node tools/run-node-tests.js tests/kernels/hyper-connection-up-contract.test.js tests/kernels/hyper-connection-up-boundary.test.js tests/inference/hyper-connection-mix-branch-wiring.test.js tests/inference/hyper-connection-layer-wiring.test.js
```

In a coordinated clean physical GPU window only:

```sh
node tests/kernels/hyper-connection-up-gpu.test.js
```

This bounded synthetic test requires exact output bits against both the original
three-dispatch chain and `q4_0_gemv` plus the prior fused-mix kernel. Twelve cases
cover immediate, recorded, prepared and repeated prepared output at ranks
32/352/320 and widths 65/257/2560. It admits at most 8 MiB of observed GPUBuffer
allocations and requires explicit zero-byte retirement before device destruction.
It is not learned-model parity or performance qualification.
