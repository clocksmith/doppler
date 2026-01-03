// Rotary Position Embeddings (RoPE) Kernel (F16)
//
// Applies rotary position embeddings to Q and K tensors.
// Same math as rope.wgsl, but input/output are f16.
// Computation is performed in f32 for stability.
//
// Supports:
// - Original RoPE (base = 10000)
// - Scaled RoPE (for extended context)
// - NTK-aware scaling

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

// Mathematical constant
const PI: f32 = 3.14159265359;

struct Uniforms {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    start_pos: u32,
    rope_base: f32,
    rope_scale: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> freqs_cos: array<f32>;
@group(0) @binding(3) var<storage, read_write> freqs_sin: array<f32>;

// Apply RoPE using precomputed frequencies
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let head_dim = u.head_dim;
    let num_heads = u.num_heads;
    let seq_len = u.seq_len;
    let start_pos = u.start_pos;

    let half_dim = head_dim / 2u;
    let total_pairs = seq_len * num_heads * half_dim;
    let idx = global_id.x;

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
    let x0 = f32(input[base_idx + pair_idx]);
    let x1 = f32(input[base_idx + pair_idx + half_dim]);

    let y0 = x0 * cos_val - x1 * sin_val;
    let y1 = x0 * sin_val + x1 * cos_val;

    input[base_idx + pair_idx] = f16(y0);
    input[base_idx + pair_idx + half_dim] = f16(y1);
}

// Compute frequencies on-the-fly (no precomputation needed)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rope_compute_freqs(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let head_dim = u.head_dim;
    let num_heads = u.num_heads;
    let seq_len = u.seq_len;
    let start_pos = u.start_pos;
    let rope_base = u.rope_base;
    let rope_scale = u.rope_scale;

    let idx = global_id.x;
    let half_dim = head_dim / 2u;
    let total_pairs = seq_len * num_heads * half_dim;

    if (idx >= total_pairs) {
        return;
    }

    let pos = idx / (num_heads * half_dim);
    let remainder = idx % (num_heads * half_dim);
    let head_idx = remainder / half_dim;
    let pair_idx = remainder % half_dim;

    let actual_pos = f32(start_pos + pos) / rope_scale;

    let exponent = f32(pair_idx * 2u) / f32(head_dim);
    let freq = 1.0 / pow(rope_base, exponent);
    let theta = actual_pos * freq;

    let cos_val = cos(theta);
    let sin_val = sin(theta);

    let base_idx = pos * num_heads * head_dim + head_idx * head_dim;
    let x0 = f32(input[base_idx + pair_idx]);
    let x1 = f32(input[base_idx + pair_idx + half_dim]);

    let y0 = x0 * cos_val - x1 * sin_val;
    let y1 = x0 * sin_val + x1 * cos_val;

    input[base_idx + pair_idx] = f16(y0);
    input[base_idx + pair_idx + half_dim] = f16(y1);
}

// Apply RoPE to both Q and K in one pass
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rope_qk(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let head_dim = u.head_dim;
    let num_heads = u.num_heads;
    let seq_len = u.seq_len;
    let start_pos = u.start_pos;
    let rope_base = u.rope_base;
    let rope_scale = u.rope_scale;

    let idx = global_id.x;
    let half_dim = head_dim / 2u;
    let total_pairs = seq_len * num_heads * half_dim;

    if (idx >= total_pairs) {
        return;
    }

    let pos = idx / (num_heads * half_dim);
    let remainder = idx % (num_heads * half_dim);
    let head_idx = remainder / half_dim;
    let pair_idx = remainder % half_dim;

    let actual_pos = f32(start_pos + pos) / rope_scale;

    let exponent = f32(pair_idx * 2u) / f32(head_dim);
    let freq = 1.0 / pow(rope_base, exponent);
    let theta = actual_pos * freq;

    let cos_val = cos(theta);
    let sin_val = sin(theta);

    let q_base_idx = pos * num_heads * head_dim * 2u + head_idx * head_dim;
    let k_base_idx = q_base_idx + head_dim;

    let q0 = f32(input[q_base_idx + pair_idx]);
    let q1 = f32(input[q_base_idx + pair_idx + half_dim]);
    input[q_base_idx + pair_idx] = f16(q0 * cos_val - q1 * sin_val);
    input[q_base_idx + pair_idx + half_dim] = f16(q0 * sin_val + q1 * cos_val);

    let k0 = f32(input[k_base_idx + pair_idx]);
    let k1 = f32(input[k_base_idx + pair_idx + half_dim]);
    input[k_base_idx + pair_idx] = f16(k0 * cos_val - k1 * sin_val);
    input[k_base_idx + pair_idx + half_dim] = f16(k0 * sin_val + k1 * cos_val);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rope_ntk_scaled(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let head_dim = u.head_dim;
    let num_heads = u.num_heads;
    let seq_len = u.seq_len;
    let start_pos = u.start_pos;
    var rope_base = u.rope_base;
    let rope_scale = u.rope_scale;

    let idx = global_id.x;
    let half_dim = head_dim / 2u;
    let total_pairs = seq_len * num_heads * half_dim;

    if (idx >= total_pairs) {
        return;
    }

    rope_base = rope_base * pow(rope_scale, f32(head_dim) / (f32(head_dim) - 2.0));

    let pos = idx / (num_heads * half_dim);
    let remainder = idx % (num_heads * half_dim);
    let head_idx = remainder / half_dim;
    let pair_idx = remainder % half_dim;

    let actual_pos = f32(start_pos + pos);

    let exponent = f32(pair_idx * 2u) / f32(head_dim);
    let freq = 1.0 / pow(rope_base, exponent);
    let theta = actual_pos * freq;

    let cos_val = cos(theta);
    let sin_val = sin(theta);

    let base_idx = pos * num_heads * head_dim + head_idx * head_dim;
    let x0 = f32(input[base_idx + pair_idx]);
    let x1 = f32(input[base_idx + pair_idx + half_dim]);

    input[base_idx + pair_idx] = f16(x0 * cos_val - x1 * sin_val);
    input[base_idx + pair_idx + half_dim] = f16(x0 * sin_val + x1 * cos_val);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rope_yarn(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let head_dim = u.head_dim;
    let num_heads = u.num_heads;
    let seq_len = u.seq_len;
    let start_pos = u.start_pos;
    let rope_base = u.rope_base;
    let rope_scale = u.rope_scale;

    let idx = global_id.x;
    let half_dim = head_dim / 2u;
    let total_pairs = seq_len * num_heads * half_dim;

    if (idx >= total_pairs) {
        return;
    }

    let pos = idx / (num_heads * half_dim);
    let remainder = idx % (num_heads * half_dim);
    let head_idx = remainder / half_dim;
    let pair_idx = remainder % half_dim;

    let actual_pos = f32(start_pos + pos);

    let beta_fast: f32 = 32.0;
    let beta_slow: f32 = 1.0;

    let exponent = f32(pair_idx * 2u) / f32(head_dim);
    let orig_freq = 1.0 / pow(rope_base, exponent);

    let wavelength = 2.0 * PI / orig_freq;

    var ramp: f32;
    let low_wavelength = f32(head_dim) / beta_fast;
    let high_wavelength = f32(head_dim) / beta_slow;

    if (wavelength < low_wavelength) {
        ramp = 0.0;
    } else if (wavelength > high_wavelength) {
        ramp = 1.0;
    } else {
        ramp = (wavelength - low_wavelength) / (high_wavelength - low_wavelength);
    }

    let scaled_pos = actual_pos / rope_scale;
    let interp_pos = (1.0 - ramp) * actual_pos + ramp * scaled_pos;

    let theta = interp_pos * orig_freq;
    let cos_val = cos(theta);
    let sin_val = sin(theta);

    let base_idx = pos * num_heads * head_dim + head_idx * head_dim;
    let x0 = f32(input[base_idx + pair_idx]);
    let x1 = f32(input[base_idx + pair_idx + half_dim]);

    input[base_idx + pair_idx] = f16(x0 * cos_val - x1 * sin_val);
    input[base_idx + pair_idx + half_dim] = f16(x0 * sin_val + x1 * cos_val);
}
