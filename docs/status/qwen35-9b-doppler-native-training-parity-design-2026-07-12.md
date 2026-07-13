# Qwen 3.5 9B Doppler-Native Training Parity Design

This is a separate post-V12 mechanics experiment. It does not change, restart,
or expand the frozen V12 data-lane matrix.

The task is completion-only WGSL repair. The student is
`Qwen/Qwen3.5-9B` at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. There is no live V12 teacher:
training targets are stored, compiler-accepted repaired WGSL spans. Gamma's
PyTorch/ROCm trainer is the qualified reference backend; Doppler's native
WebGPU/Vulkan trainer is the candidate backend.

The experiment follows Gamma SAME-R: different inner backends, matched rims.
It is not a claim that SAME-R selection is automated.

## Causal boundary

The intervention is the training backend and its declared precision bundle.
The model revision, tokenizer, source rows, row order, initial LoRA tensors,
completion mask, target modules, optimizer equations, microstep budget,
evaluation prompts, and artifact checks remain fixed.

WebGPU has no standard BF16 arithmetic path. Therefore the first numerical
oracle is not the BF16 V12 run. It is a matched F16-storage/F32-accumulation
reference in Gamma compared with Doppler's F16 frozen weights and F32
activations, gradients, LoRA parameters, and optimizer state. A later BF16
Gamma versus F16/F32 Doppler comparison is a backend-plus-precision bundle and
may establish behavioral equivalence, not tensor identity.

## Staged gates

| Gate | Matched operation | Required evidence |
| --- | --- | --- |
| F16 projection dX | Same F16 weight bytes and F32 upstream gradient | Both storage orientations finite and max absolute error at most `2e-6`; a perturbed-weight negative control must exceed `1e-4`. |
| Full-attention block | Same one-layer inputs, LoRA tensors, masks, and loss seed | Forward loss, every adapter gradient, and every updated adapter tensor compared against the reference; no missing or non-finite tensors. |
| Linear-attention block | Same Qwen gated-delta projections, convolution state, recurrence, and loss seed | Forward and backward parity for `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, short convolution, recurrent state, norm, and output projection. |
| One Qwen microstep | One frozen training row and identical initialized rank-32 adapter | All seven module families produce finite gradients; adapter update cosine at least `0.999` and relative L2 error at most `0.02`. |
| Matched prefix | Same ordered training-row prefix and AdamW state | Equal consumed-prefix hash, optimizer-step count, no non-finite values, every expected adapter tensor changed, and adapter update cosine at least `0.995`. |
| Export and inference | Same adapter tensors through PEFT and Doppler packages | Name/shape mapping complete, tensor checksum lineage complete, coherent activation, and matched held-out completions under one decoder contract. |
| Sustained candidate | Frozen training and external compiler evaluator | Only after all earlier gates; capability remains separate from mechanics and still requires semantic-kernel evaluation for promotion. |

The full-attention and linear-attention gates are deliberately separate.
Qwen 3.5 9B has 24 `linear_attention` layers and eight `full_attention`
layers. Treating all 32 layers as ordinary attention would be a different
model, not a backend port.

## Implemented first slice

Doppler commit `4ee6406f` adds:

- packed F16 frozen-weight input gradients with F32 accumulation;
- orientation coverage for both `W[K,N]` and `W[N,K]` layouts;
- stopped-gradient propagation that avoids allocating frozen base-weight and
  embedding gradients;
- preservation of frozen F16 matrix storage in the native LoRA fixture;
- an independent scalar browser oracle with a perturbed-weight negative
  control; and
- a generated kernel digest and explicit registry variant.

Static kernel, registry, autograd, and training tests pass. The numerical
browser oracle is still unexecuted because the active V12 ROCm evaluator owns
the accelerator. No kernel-mechanics pass is claimed until its JSON receipt
exists.

## Known blocking gaps

- Native Qwen gated-delta/linear-attention backward is absent.
- Native LoRA currently represents a fused `gate_up_proj`; V12 uses separate
  `gate_proj` and `up_proj` tensors.
- Gradient checkpointing is not qualified for the Qwen hybrid graph.
- A matched initial-adapter importer and PEFT export parity receipt are absent.
- Sustained AdamW accumulation and resume have not run on Qwen 9B.
- Doppler-native inference for the trained adapter remains separate from base
  F16 inference and from the rejected mixed-Q4 artifact.

Until those gaps close, `qwen-3-5-9b-hf-bf16` remains registered as requiring
an external trainer. The machine-readable design is
[qwen35-9b-doppler-native-training-parity-design-2026-07-12.json](qwen35-9b-doppler-native-training-parity-design-2026-07-12.json).
