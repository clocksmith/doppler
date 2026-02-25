import { WGSL_PATCH_VARIANTS } from './wgsl-patch-variants.js';

export const WGSL_GENERATED_VARIANTS = Object.freeze([
  {
    id: 'conv2d-f16',
    source: 'src/gpu/kernels/conv2d.wgsl',
    target: 'src/gpu/kernels/conv2d_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// Conv2D Kernel (NCHW)
//
// Naive direct convolution with padding and stride.
`,
        to: `// Conv2D Kernel (NCHW, f16)

enable f16;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> input: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> input: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(2) var<storage, read> weight: array<f32>;',
        to: '@group(0) @binding(2) var<storage, read> weight: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(3) var<storage, read> bias: array<f32>;',
        to: '@group(0) @binding(3) var<storage, read> bias: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(4) var<storage, read_write> output: array<f32>;',
        to: '@group(0) @binding(4) var<storage, read_write> output: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '    var sum: f32 = bias[out_c];',
        to: '    var sum: f32 = f32(bias[out_c]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '                sum = sum + input[input_idx] * weight[weight_idx];',
        to: '                sum = sum + f32(input[input_idx]) * f32(weight[weight_idx]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '    output[idx] = sum;',
        to: '    output[idx] = f16(sum);',
      },
    ],
  },
  {
    id: 'energy-eval-f16',
    source: 'src/gpu/kernels/energy_eval.wgsl',
    target: 'src/gpu/kernels/energy_eval_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// energy_eval.wgsl
// Computes per-element energy contributions for (state - target)^2.
`,
        to: `// energy_eval_f16.wgsl
// Computes per-element energy contributions for f16 inputs.

enable f16;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> state: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> state: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(2) var<storage, read> targetBuf: array<f32>;',
        to: '@group(0) @binding(2) var<storage, read> targetBuf: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '    let diff = state[idx] - targetBuf[idx];',
        to: '    let diff = f32(state[idx] - targetBuf[idx]);',
      },
    ],
  },
  {
    id: 'energy-update-f16',
    source: 'src/gpu/kernels/energy_update.wgsl',
    target: 'src/gpu/kernels/energy_update_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// energy_update.wgsl
// Gradient step on state towards target: state -= stepSize * gradientScale * (state - target).
`,
        to: `// energy_update_f16.wgsl
// Gradient step on f16 state towards f16 target.

enable f16;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read_write> state: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read_write> state: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(2) var<storage, read> targetBuf: array<f32>;',
        to: '@group(0) @binding(2) var<storage, read> targetBuf: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: `    let diff = state[idx] - targetBuf[idx];
    state[idx] = state[idx] - (u.stepSize * u.gradientScale * diff);`,
        to: `    let diff = f32(state[idx] - targetBuf[idx]);
    let next = f32(state[idx]) - (u.stepSize * u.gradientScale * diff);
    state[idx] = f16(next);`,
      },
    ],
  },
  {
    id: 'energy-quintel-grad-f16',
    source: 'src/gpu/kernels/energy_quintel_grad.wgsl',
    target: 'src/gpu/kernels/energy_quintel_grad_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// energy_quintel_grad.wgsl
// Quintel gradient kernel: compute dE/d(state) for symmetry/count/center/binarize terms.
`,
        to: `// energy_quintel_grad_f16.wgsl
// Quintel gradient kernel for f16 state: compute dE/d(state) into f32 output.

enable f16;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> state: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> state: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '    let value = state[idx];',
        to: '    let value = f32(state[idx]);',
      },
      {
        type: 'literal',
        count: 3,
        from: '        let diff = value - state[mirrorIdx];',
        to: '        let diff = value - f32(state[mirrorIdx]);',
      },
    ],
  },
  {
    id: 'energy-quintel-reduce-f16',
    source: 'src/gpu/kernels/energy_quintel_reduce.wgsl',
    target: 'src/gpu/kernels/energy_quintel_reduce_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// energy_quintel_reduce.wgsl
// Reduces quintel energy components and state sum per workgroup.
`,
        to: `// energy_quintel_reduce_f16.wgsl
// Reduces quintel energy components and state sum per workgroup for f16 state.

enable f16;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> state: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> state: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        let value = state[idx];',
        to: '        let value = f32(state[idx]);',
      },
      {
        type: 'literal',
        count: 3,
        from: '                let diff = value - state[mirrorIdx];',
        to: '                let diff = value - f32(state[mirrorIdx]);',
      },
    ],
  },
  {
    id: 'energy-quintel-update-f16',
    source: 'src/gpu/kernels/energy_quintel_update.wgsl',
    target: 'src/gpu/kernels/energy_quintel_update_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// energy_quintel_update.wgsl
// Quintel update kernel: apply symmetry/count/center/binarize gradients.
`,
        to: `// energy_quintel_update_f16.wgsl
// Quintel update kernel for f16 state.

enable f16;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read_write> state: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read_write> state: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '    let value = state[idx];',
        to: '    let value = f32(state[idx]);',
      },
      {
        type: 'literal',
        count: 3,
        from: '        let diff = value - state[mirrorIdx];',
        to: '        let diff = value - f32(state[mirrorIdx]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '    state[idx] = next;',
        to: '    state[idx] = f16(next);',
      },
    ],
  },
  {
    id: 'groupnorm-apply-f16',
    source: 'src/gpu/kernels/groupnorm_apply.wgsl',
    target: 'src/gpu/kernels/groupnorm_apply_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: '// GroupNorm Apply Kernel (NCHW)\n',
        to: '// GroupNorm Apply Kernel (NCHW, f16)\n\nenable f16;\n',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> input: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> input: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(3) var<storage, read> weight: array<f32>;',
        to: '@group(0) @binding(3) var<storage, read> weight: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(4) var<storage, read> bias: array<f32>;',
        to: '@group(0) @binding(4) var<storage, read> bias: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(5) var<storage, read_write> output: array<f32>;',
        to: '@group(0) @binding(5) var<storage, read_write> output: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: `    let value = (input[idx] - mean) * inv_std;
    output[idx] = value * weight[channel] + bias[channel];`,
        to: `    let value = (f32(input[idx]) - mean) * inv_std;
    let scaled = value * f32(weight[channel]) + f32(bias[channel]);
    output[idx] = f16(scaled);`,
      },
    ],
  },
  {
    id: 'groupnorm-stats-f16',
    source: 'src/gpu/kernels/groupnorm_stats.wgsl',
    target: 'src/gpu/kernels/groupnorm_stats_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// GroupNorm Stats Kernel (NCHW)
// Computes mean and inv-std for each group.
`,
        to: `// GroupNorm Stats Kernel (NCHW, f16 input)

enable f16;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> input: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> input: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        let value = input[base + idx];',
        to: '        let value = f32(input[base + idx]);',
      },
    ],
  },
  {
    id: 'modulate-f16',
    source: 'src/gpu/kernels/modulate.wgsl',
    target: 'src/gpu/kernels/modulate_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// Modulate kernel
// Applies per-channel affine and optional gating.

override WORKGROUP_SIZE: u32 = 256u;
`,
        to: `// Modulate kernel (f16)
// Applies per-channel affine and optional gating.

enable f16;
override WORKGROUP_SIZE: u32 = 256u;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> input: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> input: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(2) var<storage, read> mod_params: array<f32>;',
        to: '@group(0) @binding(2) var<storage, read> mod_params: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(3) var<storage, read_write> output: array<f32>;',
        to: '@group(0) @binding(3) var<storage, read_write> output: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '    let raw_scale = mod_params[u.scale_offset + dim];',
        to: '    let raw_scale = f32(mod_params[u.scale_offset + dim]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '    let shift = mod_params[u.shift_offset + dim];',
        to: '    let shift = f32(mod_params[u.shift_offset + dim]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '    var value = input[idx] * scale + shift;',
        to: '    var value = f32(input[idx]) * scale + shift;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        let gate = mod_params[u.gate_offset + dim];',
        to: '        let gate = f32(mod_params[u.gate_offset + dim]);',
      },
      {
        type: 'literal',
        count: 1,
        from: '    output[idx] = value;',
        to: '    output[idx] = f16(value);',
      },
    ],
  },
  {
    id: 'pixel-shuffle-f16',
    source: 'src/gpu/kernels/pixel_shuffle.wgsl',
    target: 'src/gpu/kernels/pixel_shuffle_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// Pixel shuffle (tokens -> CHW)

override WORKGROUP_SIZE: u32 = 256u;
`,
        to: `// Pixel shuffle (tokens -> CHW) f16

enable f16;
override WORKGROUP_SIZE: u32 = 256u;
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> input: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> input: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(2) var<storage, read_write> output: array<f32>;',
        to: '@group(0) @binding(2) var<storage, read_write> output: array<f16>;',
      },
    ],
  },
  {
    id: 'split-qkv-f16',
    source: 'src/gpu/kernels/split_qkv.wgsl',
    target: 'src/gpu/kernels/split_qkv_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: '// split_qkv.wgsl',
        to: '// split_qkv_f16.wgsl',
      },
      {
        type: 'literal',
        count: 1,
        from: ' * Split fused QKV output into separate Q, K, V buffers.',
        to: ' * Split fused QKV output into separate Q, K, V buffers (f16).',
      },
      {
        type: 'literal',
        count: 1,
        from: '\nstruct Params {',
        to: '\nenable f16;\n\nstruct Params {',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> input: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> input: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(2) var<storage, read_write> Q: array<f32>;',
        to: '@group(0) @binding(2) var<storage, read_write> Q: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(3) var<storage, read_write> K: array<f32>;',
        to: '@group(0) @binding(3) var<storage, read_write> K: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(4) var<storage, read_write> V: array<f32>;',
        to: '@group(0) @binding(4) var<storage, read_write> V: array<f16>;',
      },
    ],
  },
  {
    id: 'upsample2d-f16',
    source: 'src/gpu/kernels/upsample2d.wgsl',
    target: 'src/gpu/kernels/upsample2d_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: '// Upsample2D Kernel (nearest, NCHW)\n',
        to: '// Upsample2D Kernel (nearest, NCHW, f16)\n\nenable f16;\n',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> input: array<f32>;',
        to: '@group(0) @binding(1) var<storage, read> input: array<f16>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(2) var<storage, read_write> output: array<f32>;',
        to: '@group(0) @binding(2) var<storage, read_write> output: array<f16>;',
      },
    ],
  },
  {
    id: 'gather-f16',
    source: 'src/gpu/kernels/gather.wgsl',
    target: 'src/gpu/kernels/gather_f16.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: '// gather.wgsl',
        to: '// gather_f16.wgsl',
      },
      {
        type: 'literal',
        count: 1,
        from: ' * Gather Kernel - Token Embedding Lookup',
        to: ' * Gather Kernel (F16) - Token Embedding Lookup with F16 Embeddings',
      },
      {
        type: 'literal',
        count: 1,
        from: ' * Gathers rows from an embedding matrix based on token indices.',
        to: ' * Gathers rows from an F16 embedding matrix based on token indices.',
      },
      {
        type: 'literal',
        count: 1,
        from: ' * Used for efficient embedding lookup on GPU without CPU readback.',
        to: ' * Outputs F32 for downstream computation (activations are F32).',
      },
      {
        type: 'literal',
        count: 1,
        from: '\n// Tunable workgroup size',
        to: '\nenable f16;\n\n// Tunable workgroup size',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(2) var<storage, read> embeddings: array<f32>;   // Embedding matrix [vocab_size, hidden_size]',
        to: '@group(0) @binding(2) var<storage, read> embeddings: array<f16>;   // F16 Embedding matrix [vocab_size, hidden_size]',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(3) var<storage, read_write> output: array<f32>; // Output [num_tokens, hidden_size]',
        to: '@group(0) @binding(3) var<storage, read_write> output: array<f32>; // F32 Output [num_tokens, hidden_size]',
      },
      {
        type: 'literal',
        count: 1,
        from: '    // Gather from embedding matrix',
        to: '    // Gather from F16 embedding matrix, convert to F32 output',
      },
      {
        type: 'literal',
        count: 1,
        from: '    output[tid] = embeddings[embed_offset];',
        to: '    output[tid] = f32(embeddings[embed_offset]);',
      },
    ],
  },
  {
    id: 'attention-decode-online-f16kv',
    source: 'src/gpu/kernels/attention_decode_online_f16.wgsl',
    target: 'src/gpu/kernels/attention_decode_online_f16kv.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: '// Online Decode Attention Kernel (f16 QKV + f16 output)',
        to: '// Online Decode Attention Kernel (f32 Q + f16 KV + f32 output)',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> Q: array<f16>;',
        to: '@group(0) @binding(1) var<storage, read> Q: array<f32>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(4) var<storage, read_write> output: array<f16>;',
        to: '@group(0) @binding(4) var<storage, read_write> output: array<f32>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '            output[out_offset] = f16(0.0);',
        to: '            output[out_offset] = 0.0;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        shared_q[tid] = f32(Q[q_offset]);',
        to: '        shared_q[tid] = Q[q_offset];',
      },
      {
        type: 'literal',
        count: 1,
        from: '        output[out_offset] = f16(out_accum * inv_sum);',
        to: '        output[out_offset] = out_accum * inv_sum;',
      },
    ],
  },
  {
    id: 'attention-decode-paged-f16kv',
    source: 'src/gpu/kernels/attention_decode_paged_f16.wgsl',
    target: 'src/gpu/kernels/attention_decode_paged_f16kv.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: `// Paged Decode Attention Kernel (f16 Q/K/V + f16 output)
//
// Uses a page table to map logical KV positions to physical pages.
`,
        to: `// Paged Decode Attention Kernel (f16 KV)
//
// Q is f32, K/V are f16, output is f32. Uses page table indirection.
`,
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> Q: array<f16>;',
        to: '@group(0) @binding(1) var<storage, read> Q: array<f32>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(4) var<storage, read_write> output: array<f16>;',
        to: '@group(0) @binding(4) var<storage, read_write> output: array<f32>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        q_val = f32(Q[q_offset]);',
        to: '        q_val = Q[q_offset];',
      },
      {
        type: 'literal',
        count: 1,
        from: '        output[out_offset] = f16(shared_acc[tid]);',
        to: '        output[out_offset] = shared_acc[tid];',
      },
    ],
  },
  {
    id: 'attention-decode-tiered-f16kv',
    source: 'src/gpu/kernels/attention_decode_tiered_f16.wgsl',
    target: 'src/gpu/kernels/attention_decode_tiered_f16kv.wgsl',
    rules: [
      {
        type: 'literal',
        count: 1,
        from: '// Tiered Decode Attention Kernel (f16 QKV + f16 output)',
        to: '// Tiered Decode Attention Kernel (f32 Q + f16 KV)',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(1) var<storage, read> Q: array<f16>;',
        to: '@group(0) @binding(1) var<storage, read> Q: array<f32>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '@group(0) @binding(6) var<storage, read_write> output: array<f16>;',
        to: '@group(0) @binding(6) var<storage, read_write> output: array<f32>;',
      },
      {
        type: 'literal',
        count: 1,
        from: '        q_val = f32(Q[q_offset]);',
        to: '        q_val = Q[q_offset];',
      },
      {
        type: 'literal',
        count: 1,
        from: '        output[out_offset] = f16(shared_acc[tid]);',
        to: '        output[out_offset] = shared_acc[tid];',
      },
    ],
  },
  ...WGSL_PATCH_VARIANTS,
]);
