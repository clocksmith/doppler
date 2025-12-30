// Rotary Position Embeddings (RoPE) Kernel
//
// Applies rotary position embeddings to Q and K tensors.
// RoPE rotates pairs of dimensions based on position and frequency.
//
// For each pair (x_i, x_{i+1}):
//   x'_i     = x_i * cos(θ) - x_{i+1} * sin(θ)
//   x'_{i+1} = x_i * sin(θ) + x_{i+1} * cos(θ)
//
// Where θ = pos * freq_i, freq_i = 1 / (base^(2i/d))
//
// Supports:
// - Original RoPE (base = 10000)
// - Scaled RoPE (for extended context)
// - NTK-aware scaling

override WORKGROUP_SIZE: u32 = 256u;

// Mathematical constant
const PI: f32 = 3.14159265359;

struct Uniforms {
    seq_len: u32,          // Sequence length
    num_heads: u32,        // Number of heads
    head_dim: u32,         // Dimension per head (must be even)
    start_pos: u32,        // Starting position (for decode)
    rope_base: f32,        // Base frequency (default 10000)
    rope_scale: f32,       // Scaling factor for extended context
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;  // [seq_len, num_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> freqs_cos: array<f32>;  // Precomputed cos [max_seq_len, head_dim/2]
@group(0) @binding(3) var<storage, read_write> freqs_sin: array<f32>;  // Precomputed sin [max_seq_len, head_dim/2]

// Apply RoPE using precomputed frequencies
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let head_dim = u.head_dim;
    let num_heads = u.num_heads;
    let seq_len = u.seq_len;
    let start_pos = u.start_pos;

    // Global thread index (one thread per complex pair)
    let half_dim = head_dim / 2u;
    let total_pairs = seq_len * num_heads * half_dim;
    let idx = global_id.x;

    if (idx >= total_pairs) {
        return;
    }

    // Decompose index
    let pos = idx / (num_heads * half_dim);
    let remainder = idx % (num_heads * half_dim);
    let head_idx = remainder / half_dim;
    let pair_idx = remainder % half_dim;
    let actual_pos = start_pos + pos;

    // Get precomputed cos/sin for this position and dimension
    let freq_idx = actual_pos * half_dim + pair_idx;
    let cos_val = freqs_cos[freq_idx];
    let sin_val = freqs_sin[freq_idx];

    // Apply "rotate-half" layout: pair (x[i], x[i + half_dim])
    let base_idx = pos * num_heads * head_dim + head_idx * head_dim;
    let x0 = input[base_idx + pair_idx];
    let x1 = input[base_idx + pair_idx + half_dim];

    // Apply rotation
    let y0 = x0 * cos_val - x1 * sin_val;
    let y1 = x0 * sin_val + x1 * cos_val;

    // Write back
    input[base_idx + pair_idx] = y0;
    input[base_idx + pair_idx + half_dim] = y1;
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

    // Decompose index
    let pos = idx / (num_heads * half_dim);
    let remainder = idx % (num_heads * half_dim);
    let head_idx = remainder / half_dim;
    let pair_idx = remainder % half_dim;

    let actual_pos = f32(start_pos + pos) / rope_scale;

    // Compute frequency: 1 / (base^(2*pair_idx/head_dim))
    let exponent = f32(pair_idx * 2u) / f32(head_dim);
    let freq = 1.0 / pow(rope_base, exponent);
    let theta = actual_pos * freq;

    let cos_val = cos(theta);
    let sin_val = sin(theta);

    // Apply "rotate-half" layout: pair (x[i], x[i + half_dim])
    let base_idx = pos * num_heads * head_dim + head_idx * head_dim;
    let x0 = input[base_idx + pair_idx];
    let x1 = input[base_idx + pair_idx + half_dim];

    // Apply rotation
    input[base_idx + pair_idx] = x0 * cos_val - x1 * sin_val;
    input[base_idx + pair_idx + half_dim] = x0 * sin_val + x1 * cos_val;
}

// Apply RoPE to both Q and K in one pass
// Input layout: Q and K concatenated [seq_len, num_heads, head_dim * 2]
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
    // Each thread handles one Q-K pair at one dimension pair
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

    // Compute frequency
    let exponent = f32(pair_idx * 2u) / f32(head_dim);
    let freq = 1.0 / pow(rope_base, exponent);
    let theta = actual_pos * freq;

    let cos_val = cos(theta);
    let sin_val = sin(theta);

    // Q is in first half, K in second half
    let q_base_idx = pos * num_heads * head_dim * 2u + head_idx * head_dim;
    let k_base_idx = q_base_idx + head_dim;  // K starts after Q

    // Process Q
    let q0 = input[q_base_idx + pair_idx];
    let q1 = input[q_base_idx + pair_idx + half_dim];
    input[q_base_idx + pair_idx] = q0 * cos_val - q1 * sin_val;
    input[q_base_idx + pair_idx + half_dim] = q0 * sin_val + q1 * cos_val;

    // Process K
    let k0 = input[k_base_idx + pair_idx];
    let k1 = input[k_base_idx + pair_idx + half_dim];
    input[k_base_idx + pair_idx] = k0 * cos_val - k1 * sin_val;
    input[k_base_idx + pair_idx + half_dim] = k0 * sin_val + k1 * cos_val;
}

// Precompute frequency table (run once at init)
// Output: freqs_cos, freqs_sin [maxSeqLen, head_dim/2]
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn precompute_freqs(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let head_dim = u.head_dim;
    let seq_len = u.seq_len;  // maxSeqLen for precomputation
    let rope_base = u.rope_base;
    let rope_scale = u.rope_scale;

    let idx = global_id.x;
    let half_dim = head_dim / 2u;
    let total_elements = seq_len * half_dim;

    if (idx >= total_elements) {
        return;
    }

    let pos = idx / half_dim;
    let dim_idx = idx % half_dim;

    let actual_pos = f32(pos) / rope_scale;
    let exponent = f32(dim_idx * 2u) / f32(head_dim);
    let freq = 1.0 / pow(rope_base, exponent);
    let theta = actual_pos * freq;

    freqs_cos[idx] = cos(theta);
    freqs_sin[idx] = sin(theta);
}

// NTK-aware scaled RoPE (for extended context without fine-tuning)
// Uses dynamic scaling based on sequence length
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

    // NTK scaling: increase base proportionally to scale factor
    // This preserves high-frequency components better than linear interpolation
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
    let x0 = input[base_idx + pair_idx];
    let x1 = input[base_idx + pair_idx + half_dim];

    input[base_idx + pair_idx] = x0 * cos_val - x1 * sin_val;
    input[base_idx + pair_idx + half_dim] = x0 * sin_val + x1 * cos_val;
}

// YaRN-style RoPE with attention scaling
// Combines NTK interpolation with linear interpolation based on frequency
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

    // YaRN parameters
    let beta_fast: f32 = 32.0;
    let beta_slow: f32 = 1.0;
    let alpha: f32 = 1.0;

    // Compute original frequency
    let exponent = f32(pair_idx * 2u) / f32(head_dim);
    let orig_freq = 1.0 / pow(rope_base, exponent);

    // Compute wavelength
    let wavelength = 2.0 * PI / orig_freq;

    // Interpolation factor based on wavelength
    var ramp: f32;
    let low_wavelength = f32(head_dim) / beta_fast;
    let high_wavelength = f32(head_dim) / beta_slow;

    if (wavelength < low_wavelength) {
        ramp = 0.0;  // No interpolation for high frequencies
    } else if (wavelength > high_wavelength) {
        ramp = 1.0;  // Full interpolation for low frequencies
    } else {
        ramp = (wavelength - low_wavelength) / (high_wavelength - low_wavelength);
    }

    // Combine original and scaled position
    let scaled_pos = actual_pos / rope_scale;
    let interp_pos = (1.0 - ramp) * actual_pos + ramp * scaled_pos;

    let theta = interp_pos * orig_freq;
    let cos_val = cos(theta);
    let sin_val = sin(theta);

    let base_idx = pos * num_heads * head_dim + head_idx * head_dim;
    let x0 = input[base_idx + pair_idx];
    let x1 = input[base_idx + pair_idx + half_dim];

    input[base_idx + pair_idx] = x0 * cos_val - x1 * sin_val;
    input[base_idx + pair_idx + half_dim] = x0 * sin_val + x1 * cos_val;
}
