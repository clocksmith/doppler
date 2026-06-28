// Rotary Position Embeddings for separate f16 Q/K buffers.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    rotary_dim: u32,
    interleaved: u32,
    pair_span_dim: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> q_input: array<f16>;
@group(0) @binding(2) var<storage, read_write> k_input: array<f16>;
@group(0) @binding(3) var<storage, read> freqs_cos: array<f32>;
@group(0) @binding(4) var<storage, read> freqs_sin: array<f32>;

fn get_first_rotary_idx(pair_idx: u32) -> u32 {
    if (u.interleaved == 1u) {
        return pair_idx * 2u;
    }
    return pair_idx;
}

fn get_second_rotary_idx(pair_idx: u32, pair_span_dim: u32) -> u32 {
    if (u.interleaved == 1u) {
        return pair_idx * 2u + 1u;
    }
    return pair_idx + (pair_span_dim / 2u);
}

fn rotate_pair(x0: f16, x1: f16, cos_val: f16, sin_val: f16) -> vec2<f16> {
    return vec2<f16>(
        x0 * cos_val - x1 * sin_val,
        x0 * sin_val + x1 * cos_val
    );
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let half_dim = u.rotary_dim / 2u;
    let q_pairs = u.seq_len * u.num_q_heads * half_dim;
    let k_pairs = u.seq_len * u.num_kv_heads * half_dim;
    let total_pairs = q_pairs + k_pairs;
    let idx = global_id.x;

    if (idx >= total_pairs) {
        return;
    }

    if (idx < q_pairs) {
        let pos = idx / (u.num_q_heads * half_dim);
        let remainder = idx % (u.num_q_heads * half_dim);
        let head_idx = remainder / half_dim;
        let pair_idx = remainder % half_dim;
        let freq_idx = (u.start_pos + pos) * half_dim + pair_idx;
        let cos_val = f16(freqs_cos[freq_idx]);
        let sin_val = f16(freqs_sin[freq_idx]);
        let base_idx = pos * u.num_q_heads * u.head_dim + head_idx * u.head_dim;
        let first_idx = get_first_rotary_idx(pair_idx);
        let second_idx = get_second_rotary_idx(pair_idx, u.pair_span_dim);
        let rotated = rotate_pair(
            q_input[base_idx + first_idx],
            q_input[base_idx + second_idx],
            cos_val,
            sin_val
        );
        q_input[base_idx + first_idx] = rotated.x;
        q_input[base_idx + second_idx] = rotated.y;
    } else {
        let k_idx = idx - q_pairs;
        let pos = k_idx / (u.num_kv_heads * half_dim);
        let remainder = k_idx % (u.num_kv_heads * half_dim);
        let head_idx = remainder / half_dim;
        let pair_idx = remainder % half_dim;
        let freq_idx = (u.start_pos + pos) * half_dim + pair_idx;
        let cos_val = f16(freqs_cos[freq_idx]);
        let sin_val = f16(freqs_sin[freq_idx]);
        let base_idx = pos * u.num_kv_heads * u.head_dim + head_idx * u.head_dim;
        let first_idx = get_first_rotary_idx(pair_idx);
        let second_idx = get_second_rotary_idx(pair_idx, u.pair_span_dim);
        let rotated = rotate_pair(
            k_input[base_idx + first_idx],
            k_input[base_idx + second_idx],
            cos_val,
            sin_val
        );
        k_input[base_idx + first_idx] = rotated.x;
        k_input[base_idx + second_idx] = rotated.y;
    }
}
