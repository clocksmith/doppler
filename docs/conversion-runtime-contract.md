# Conversion Runtime Contract

Canonical contract for conversion-time fields vs runtime-overridable fields.

Use this file as the source of truth when deciding whether a setting requires
re-conversion or can be changed per-run.

## Scope

- Conversion input: `request.convertPayload.converterConfig`
- Manifest output: `manifest.json`
- Runtime overlay: `runtime.inference.*`

## Single Sources of Truth

- Converter schema defaults: `src/config/schema/converter.schema.js`
- Conversion behavior: `src/converter/core.js`, `src/converter/quantization-info.js`
- Kernel-path and execution runtime behavior:
  - `docs/rdrr-format.md` (artifact contract)
  - `docs/style/config-style-guide.md` (runtime overlay rules)
  - `src/inference/pipelines/text/execution-v1.js` (strict runtime compile)

## Field Matrix

| Field | Contract owner | Persisted in manifest | Runtime-overridable | Re-convert required to change |
| --- | --- | --- | --- | --- |
| explicit conversion config fields | Conversion | Direct (`inference.*` authored in config) | No | Yes |
| `quantization.weights` | Conversion | Yes (`quantization`, tensor dtypes/layout) | No | Yes |
| `quantization.embeddings` | Conversion | Yes | No | Yes |
| `quantization.lmHead` | Conversion | Yes | No | Yes |
| `artifactIdentity.sourceCheckpointId` | Conversion/catalog migration | Yes | No | No, if correcting metadata only |
| `artifactIdentity.weightCapsuleId` | Conversion/catalog migration | Yes | No | Yes, when converted tensor bytes change |
| `artifactIdentity.manifestVariantId` | Conversion/catalog migration | Yes | No | No, if only manifest/runtime policy changes over the same weight capsule |
| `weightsRef` | Manifest variant migration | Yes | No | No, if redirecting a manifest variant to an existing weight capsule |
| `output.textOnly` | Conversion | Indirect (tensor set emitted) | No | Yes |
| `manifest.hashAlgorithm` | Conversion | Yes (`hashAlgorithm`, shard hashes) | No | Yes |
| `quantization.computePrecision` | Conversion-authored runtime default | Yes (`quantizationInfo.compute`) | Yes (through runtime/session policy) | No (for runtime behavior) |
| `inference.session.decodeLoop.*` | Conversion-authored batching policy | Yes | Via `runtime.inference.session` | No (for runtime behavior) |
| `inference.session.perLayerInputs.*` | Conversion-authored PLE materialization/cache policy | Yes | Via `runtime.inference.session` | No (for runtime behavior) |
| `output.fast` | Reserved converter config flag | No active effect in current converter path | n/a | n/a |

Notes:

- Model ID suffixes are naming only; runtime policy is driven by manifest/runtime config.
- `modelId` is release/catalog identity. It is not sufficient to identify source
  checkpoint bytes, converted Doppler shards, or manifest/runtime-policy
  variants.
- During the artifact identity migration, `artifactIdentity` and `weightsRef`
  are additive metadata. Runtime shard resolution must not silently use
  `weightsRef` until an explicit loader contract is implemented.
- Storage dtype changes always require re-conversion.
- Direct-source runtime adaptation builds runtime-model contracts for `safetensors`, `gguf`, and `.tflite`.
- Direct-source runtime planning is not conversion planning. It derives runtime model/session/execution state from source artifact facts plus source-runtime defaults, and it must reject converter-style quantization overrides.
- `.tflite` direct-source adaptation requires sibling `config.json` metadata.
- `.task` direct-source inputs may be either container archives or raw TFLite FlatBuffers. Runtime detection must classify the artifact explicitly and fail closed on unknown task packaging.
- Float TFLite constants are loaded directly.
- Quantized TFLite constants currently support per-tensor affine `INT8`, `UINT8`, and `INT4` source weights via explicit `tensor.sourceTransform` runtime-model metadata, dequantized at load time to `F16`.
- LiteRT/TFLite external buffer `offset/size` tables are supported when the constant tensor contract is otherwise compatible.
- Per-channel, packed LiteRT-LM companion-tensor quantization (for example `*.w` plus `*.w_quantized_scale` / `*.sum_i`), or otherwise unsupported LiteRT/TFLite quantization must fail closed.
- Runtime overlay for execution is strict: only `runtime.inference.session` is accepted.
- `session.perLayerInputs` is the contract surface for PLE runtime policy:
  - `materialization`: `auto | range_backed | cpu_resident | gpu_resident | gpu_split_tables`
  - `rowCache.mode`: `off | lru`
  - `prefetch.mode`: `off | next_token`
  - `gpuUpload.mode`: `per_step_slices | per_batch_slices`
  - `hotCache.mode`: `off | prepared_tokens | tokenizer_scores`
  - Converter authors the baseline policy in the manifest; runtime may override it per run.

## W8A8 linear projections

The versioned numerical contract lives in
[`w8a8.schema.js`](../src/config/schema/w8a8.schema.js). Conversion must explicitly
declare it in `quantization.w8a8` and retain it in `quantizationInfo.w8a8`.
The distinct `W8A8` tensor encoding is not GGUF Q8_0, reconstructed-F32 activation
quantization, or the existing WNA8O8 metadata. Runtime cannot reinterpret one as
another.

The initial scheme quantizes attention and feed-forward linear projection
weights per output channel and their activations per token. Other tensors retain
their declared precision. Each row contains a little-endian F32 scale followed
by signed two's-complement bytes in logical reduction order, padded with zero
bytes to a multiple of four. Explicit storage descriptors bind logical shape,
row stride, scale offset and data offset. Codes range from -127 to 127; -128 is
invalid. Reduction widths above 133,144 reject before dispatch so every possible
partial sum fits I32.

Inputs are F32, finite and normal or signed zero. Nonzero subnormal inputs reject;
all-zero rows use scale one. The scale is max-absolute-value divided by 127,
rounded to nearest-even F32, and must be positive normal. Codes round the exact
ratio of the input to that stored scale to nearest-even integer, then clip to
the declared range. Rounding the quotient to F32 first is a different algorithm.
WGSL performs activation reduction and packing without a host tensor round trip.

Rescaling first converts the I32 accumulator to nearest-even F32, multiplies by
the activation scale and materializes F32, then multiplies by the weight scale
and materializes F32. Each multiplication rounds to nearest-even; each subnormal
result becomes signed zero, and overflow rejects. Native WGSL conversion and
arithmetic do not universally promise this rounding mode, so the W8A8 shaders
implement these narrow rounding operations with integer bit arithmetic.
[WGSL floating-point rules](https://www.w3.org/TR/WGSL/#floating-point-accuracy).

Portable and packed-dot kernels consume identical packed operands. The packed
implementation requires the host WGSL language feature
`packed_4x8_integer_dot_product`. Its availability does not establish faster
execution. The quantizer and both rescale specializations belong in the existing
`execution.mechanismKernels` closure with explicit shader identities and constants;
the projection's registry metadata declares those dependencies. Recorded execution
must validate GPU status before exposing results and settle owned resources on
rejection.

The [frozen qualification fixture](../tests/fixtures/w8a8-qualification.json)
separates exact operator correctness, model approximation quality, lifecycle and
complete prefill/decode costs against Q4 and F16. It does not replace the
distributed-chat reference or its 0.001 tolerance. Support requires passing
scoped installed-package and physical execution evidence, not this format's
presence in source.

## Runtime Precedence

Kernel-path resolution (low to high):

1. `runtime.inference.kernelPath`
2. per-run pipeline context override (internal runner context)

`null` is a valid "no explicit kernel path" result. Run must not invent an
implicit `'auto'` kernel path.

Execution compile order:

1. Start from `manifest.inference.session` and `manifest.inference.execution`
2. Expand compact tuples into resolved steps
3. Build inline kernel path and layer pipeline from resolved steps
4. Merge `session` into `runtime.inference.session`

Execution dtype ownership:

- config-selected kernel paths must already match runtime `activationDtype`
  and `kvDtype`
- manifest/model-selected kernel paths may seed those runtime dtypes only when
  the resolved session does not already specify conflicting values; conflicts fail closed

## Why this file exists

Other docs may describe subsets of this behavior for local context, but they
should link here rather than redefine the contract in multiple places.

## Retained rotary frequencies

`sourceRotaryFrequencies` is an optional SafeTensors conversion policy with an
explicit tensor-name pattern and expected match count. Conversion requires
contiguous positive finite f32 vectors and exact agreement across matched layers.
It retains their values as `inference.rope.ropeInverseFrequencies`; conflicting
explicit config values, missing tensors, truncated bytes, other encodings, or
per-layer differences fail before artifact emission. This source policy is not
available for GGUF or diffusion conversion.

The manifest field owns the exact global inverse-frequency vector. Runtime
overrides are rejected. The declared GPU precompute program consumes those
values, and cache identity includes them. ModelIR retains them in
`rope.inverseFrequencies`. Existing manifests without the additive field retain
the schema's null value and declared theta-generated behavior. Re-convert and
requalify an artifact to preserve source buffers that an earlier conversion
ignored; existing signed artifacts are never rewritten during opening.

The source-frequency specialization is supplied only for a non-null vector.
Theta-generated execution leaves the shader's default path selected, allowing
older signed shaders without that specialization to remain executable. Their
sealed WGSL is not replaced during opening. The physical GPU regression retains
the exact pre-change shader as its compatibility input.

## Linear-attention norm gate

`inference.normalization.linearNormGateActivation` (`silu` | `sigmoid`) is the
activation that a gated-delta layer's gated RMSNorm applies to `z`:
`norm(x) * w * act(z)`. The field is source-owned, so runtime and call-time
overrides are rejected. Manifests and Capsules converted before the field
existed omit it. Merge resolves the omission to the schema default `silu` in
`src/config/schema/manifest.schema.js` and in no other place; their converted
bytes and pinned kernels are unchanged.

Source semantics come from the upstream family, not from `output_gate_type`
alone:

- `qwen3_5` / `qwen3_5_text` hardcode silu (`Qwen3_5RMSNormGated`). A
  contradicting `output_gate_type` fails conversion.
- `qwen4_exp` / `qwen4_exp_text` apply `output_gate_type or hidden_act`, which
  must be `sigmoid` or `silu` (Flash-Next: `sigmoid`).

Conversion fails when an authored value disagrees with the source, when a
non-silu source omits the field, or when an unknown family with
linear-attention layers omits it. In both families the HF `output_gate_type`
is not the full-attention output gate, which is always sigmoid. The
`attention.outputGateType` values already recorded in checked-in configs are
not read at runtime.

Run selects the gate kernels through `kernels.gatedDeltaRecurrent` and
`kernels.gatedDeltaFusedDecode` rules. Silu keeps the original pinned shaders.
Sigmoid selects the `*_sigmoid_gate` kernels and `gated_delta_gate_sigmoid_paired`,
which differ from their silu sources only in the gate call. Every
gate-applying registry variant declares `variantMetadata.normGateActivation`.
Dispatch fails when the selected kernel's declared activation disagrees with
the model's. Execution-plan compile, paired admission and Rig signing fail
when the execution graph declares a kernel whose registry activation differs
from the manifest's. A sigmoid paired closure declares
`gated_delta_recurrent/paired_gate_sigmoid` instead of the silu gate and the
silu monolithic anchor.

## Paired Q/K head normalization steps

A paired (two-F32) graph declares per-head query and key normalization as layer
steps, `["q_norm", "<kernel key>"]` and `["k_norm", "<kernel key>"]`, placed
after the projections. The key names any registry RMSNorm program with paired
input and output, for example `rmsnorm/paired_sliced` through the graph's
`norm` key, or `rmsnorm/paired`. Run selects the program from the step for each
role, phase and layer, exactly as for the other norm roles. Resident pipeline
preparation enumerates it through the same resolver, so the prepared set equals
the dispatched set.

Run admission binds exactly one paired program to each step of every
full-attention layer in both phases. It fails closed when:

- a step names a scalar program;
- a phase lacks a step;
- a scalar session declares the steps.

Admission and Rig signing both refuse a graph that declares the steps when the
manifest normalizes no heads. Paired Q/K normalization outside these steps fails
before dispatch. That includes projection-axis and LayerNorm head normalization.

**Interim rule.** Capsules signed before these steps existed declare neither step,
such as manifests `6b2cb026` and `548e6559`. Their head norms keep running
`rmsnorm/paired`, the program those Capsules qualified. Admission and Rig signing
accept such a graph only if its closure declares that program, either as a step
kernel or in `execution.mechanismKernels`. Manifest `548e6559` declares it as
`paired_qk_head_norm`. A graph whose closure omits it is refused before loading.
Manifest `64d74719` failed this way during preparation. Remove the rule and
`UNDECLARED_PAIRED_QK_NORM_VARIANT` when every supported manifest declares the
steps. A manifest that declares the steps needs no `paired_qk_head_norm`
declaration.

## Explicit affine LayerNorm in layer plans

A layer plan may declare `op: "layernorm"` with `weight` selecting `input`,
`post_attn`, `pre_ffn`, or `post_ffn`. The corresponding affine scale and bias
must both exist. The execution-v1 tuple form is
`["layernorm", "<pinned kernel key>", "<affine selector>"]`; lowering preserves
that selector in the layer plan. Missing parameters fail rather than synthesizing
identity scale or zero bias. Pre/post-FFN bias tensors use the canonical
`pre_feedforward_layernorm.bias` and `post_feedforward_layernorm.bias` names.

LayerNorm consumes the declared normalization epsilon and preserves input dtype.
Dtype changes require a separate cast. A residual addition must be an explicit
preceding `residual_add` step, preserving source order; LayerNorm does not accept
a fused `residual` operand. Existing `rmsnorm` plans retain their semantics.

This additive operation requires a newly prepared and qualified execution plan;
it does not rewrite signed artifacts or establish BERT/MiniLM support. Immediate
and recorded execution use the existing LayerNorm WGSL implementation. Failure
cleanup retains borrowed inputs and releases owned current-state and temporary
buffers, with recorded buffers retained until recorder cleanup permits reuse.
