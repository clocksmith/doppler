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
