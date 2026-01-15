/**
 * RoPE backward kernel.
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    start_pos: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> freqs_cos: array<f32>;
@group(0) @binding(3) var<storage, read> freqs_sin: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_dim = u.head_dim;
    let num_heads = u.num_heads;
    let seq_len = u.seq_len;
    let start_pos = u.start_pos;
    let half_dim = head_dim / 2u;
    let total_pairs = seq_len * num_heads * half_dim;

    let idx = gid.x;
    if (idx >= total_pairs) {
        return;
    }

    let pos = idx / (num_heads * half_dim);
    let remainder = idx % (num_heads * half_dim);
    let head_idx = remainder / half_dim;
    let pair_idx = remainder % half_dim;
    let actual_pos = start_pos + pos;

    let freq_idx = actual_pos * half_dim + pair_idx;
    let cos_val = freqs_cos[freq_idx];
    let sin_val = freqs_sin[freq_idx];

    let base_idx = pos * num_heads * head_dim + head_idx * head_dim;
    let dy0 = grad_output[base_idx + pair_idx];
    let dy1 = grad_output[base_idx + pair_idx + half_dim];

    let dx0 = dy0 * cos_val + dy1 * sin_val;
    let dx1 = -dy0 * sin_val + dy1 * cos_val;

    output[base_idx + pair_idx] = dx0;
    output[base_idx + pair_idx + half_dim] = dx1;
}
