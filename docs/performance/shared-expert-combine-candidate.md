# Shared expert combine candidate

Component: `doppler.runtime-source.gpu.kernels` and `doppler.runtime-source.inference.pipelines.text`
Intent: preserved. Boundary effects: registered GPU arithmetic and explicit text execution identity.

The candidate replaces scalar gate repeat, sigmoid multiplication and routed-add
with one F32 `sigmoid_gated_add` dispatch. It retains the existing clamped,
refined sigmoid arithmetic and optional SwiGLU product clamp. It does not add
the outer model residual. Weight projections and routed expert arithmetic do
not change. Three dispatches become one; this does not imply a threefold speedup.

The existing shared-expert composition remains selected unless the exact
phase/layer kernel path declares:

```json
{ "op": "moe_shared_combine", "kernel": "sigmoid_gated_add_f32.wgsl", "entry": "main" }
```

No default profile, model manifest or qualification gate is changed. The call
must supply an explicit `phase` and `layerIdx` whenever a kernel path is used;
shape does not determine phase. The registered shader consumes explicit model
hidden width and device dispatch bounds. Unsupported dtypes and unresolved
identities fail before encoding. Output ownership uses `withKernelOutput`;
prepared output buffers remain workspace-owned, and recorded failure cleanup
retains newly owned output until recorder settlement.

CPU boundary/selection tests, registry/uniform/codegen/digest checks, strict
source/declaration types, style, architecture and export parity pass against
5d2c13bf. The phase/layer selection cache is scoped by immutable kernel-path
and kernel-registry identities; changing registry scope revalidates admission.

mac-host Node22.20.0 with existing webgpu0.4.0 selected observed Apple M3 Metal3.
Twelve physical immediate/recorded cases (one token, 32/33/53 tokens, hidden
widths1/7/2048, saturated scalar gates and null/positive/zero/negative clamp)
match the existing three-dispatch composition element for element: maxAbs=0.
These are component correctness receipts, not amd-host Chrome qualification.
The default-sandbox adapter-unavailable attempt and a missing-helper shader
compile failure are retained separately; neither ran numerical cases. The
shader helper omission was corrected by retaining existing SiLU refined helpers.

Pending: full-model multi-step logit/token gates and balanced production timing
on amd-host Chrome. Do not promote this candidate on dispatch count alone.
