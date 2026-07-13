// rope_backward.wgsl

/**
 * RoPE backward kernel.
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    start_pos: u32,
    rotary_dim: u32,
    pair_span_dim: u32,
    interleaved: u32,
    _pad0: u32,
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
    let idx = gid.x;
    let total_elements = seq_len * num_heads * head_dim;
    if (idx >= total_elements) {
        return;
    }
    let dim = idx % head_dim;
    let row = idx / head_dim;
    let head_idx = row % num_heads;
    let pos = row / num_heads;
    let half_dim = u.rotary_dim / 2u;
    var pair_idx = 0u;
    var is_first = false;
    var is_second = false;
    if (u.interleaved == 1u) {
        if (dim < u.rotary_dim) {
            pair_idx = dim / 2u;
            is_first = dim % 2u == 0u;
            is_second = !is_first;
        }
    } else {
        let second_start = u.pair_span_dim / 2u;
        if (dim < half_dim) {
            pair_idx = dim;
            is_first = true;
        } else if (dim >= second_start && dim < second_start + half_dim) {
            pair_idx = dim - second_start;
            is_second = true;
        }
    }
    if (!is_first && !is_second) {
        output[idx] = grad_output[idx];
        return;
    }
    let actual_pos = start_pos + pos;
    let freq_idx = actual_pos * half_dim + pair_idx;
    let cos_val = freqs_cos[freq_idx];
    let sin_val = freqs_sin[freq_idx];
    let base_idx = pos * num_heads * head_dim + head_idx * head_dim;
    let first_dim = select(pair_idx, pair_idx * 2u, u.interleaved == 1u);
    let second_dim = select(
        pair_idx + (u.pair_span_dim / 2u),
        pair_idx * 2u + 1u,
        u.interleaved == 1u
    );
    let dy0 = grad_output[base_idx + first_dim];
    let dy1 = grad_output[base_idx + second_dim];
    output[idx] = select(
        -dy0 * sin_val + dy1 * cos_val,
        dy0 * cos_val + dy1 * sin_val,
        is_first
    );
}
