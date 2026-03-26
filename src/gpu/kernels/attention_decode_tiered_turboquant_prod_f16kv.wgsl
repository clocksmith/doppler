// Tiered Decode Attention Kernel (TURBOQUANTprod cold, f16 hot, f32 output)
//
// Cold tier uses two-stage dequant:
//   Stage 1: MSE codebook dequant (b-1 bits)
//   Stage 2: 1-bit QJL residual correction
// This provides unbiased inner product estimates.

enable f16;

const MAX_KV_LEN: u32 = 2048u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_HEAD_DIM: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;
override BIT_WIDTH_MSE: u32 = 3u;
override PACK_FACTOR_MSE: u32 = 10u;
override NUM_CENTROIDS_MSE: u32 = 8u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cold_len: u32,
    hot_len: u32,
    seq_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,
    attn_softcap: f32,
    sliding_window: u32,
    hot_window: u32,
    hot_start: u32,
    packed_stride_mse: u32,
    packed_stride_residual: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K_hot: array<f16>;
@group(0) @binding(3) var<storage, read> V_hot: array<f16>;
// MSE stage:
@group(0) @binding(4) var<storage, read> K_cold_mse: array<u32>;
@group(0) @binding(5) var<storage, read> V_cold_mse: array<u32>;
@group(0) @binding(6) var<storage, read> scales_k: array<f16>;
@group(0) @binding(7) var<storage, read> scales_v: array<f16>;
// QJL residual stage:
@group(0) @binding(8) var<storage, read> residual_k: array<u32>;
@group(0) @binding(9) var<storage, read> residual_v: array<u32>;
@group(0) @binding(10) var<storage, read> residual_norms_k: array<f16>;
@group(0) @binding(11) var<storage, read> residual_norms_v: array<f16>;
// Shared:
@group(0) @binding(12) var<storage, read_write> output: array<f32>;
@group(0) @binding(13) var<storage, read> rotation_matrix: array<f32>;
@group(0) @binding(14) var<storage, read> codebook_centroids: array<f32>;
@group(0) @binding(15) var<storage, read> qjl_matrix: array<f32>;

var<workgroup> shared_scores: array<f32, MAX_KV_LEN>;
var<workgroup> shared_partial: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> shared_acc: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_rot_q: array<f32, MAX_HEAD_DIM>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    return query_head_idx / (u.num_heads / u.num_kv_heads);
}

fn is_masked(abs_key: u32) -> bool {
    let abs_query = u.start_pos;
    if (u.is_causal != 0u && abs_key > abs_query) { return true; }
    if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
        if (abs_key < abs_query - u.sliding_window + 1u) { return true; }
    }
    return false;
}

fn get_hot_pos(hot_pos: u32) -> u32 {
    if (u.hot_window > 0u) { return (u.hot_start + hot_pos) % u.hot_window; }
    return hot_pos;
}

fn unpack_mse_index(packed_val: u32, lane: u32) -> u32 {
    let bit_mask = (1u << BIT_WIDTH_MSE) - 1u;
    return (packed_val >> (BIT_WIDTH_MSE * lane)) & bit_mask;
}

fn unpack_sign_bit(packed_val: u32, bit_idx: u32) -> f32 {
    return select(-1.0, 1.0, ((packed_val >> bit_idx) & 1u) == 1u);
}

// Dequant cold K: MSE centroid * norm + QJL residual correction
fn dequant_cold_k_prod(key_pos: u32, kv_head_idx: u32, tid: u32) -> f32 {
    // MSE stage
    let pack_idx = tid / PACK_FACTOR_MSE;
    if (pack_idx >= u.packed_stride_mse) { return 0.0; }
    let lane = tid - pack_idx * PACK_FACTOR_MSE;
    let base_mse = (key_pos * u.num_kv_heads + kv_head_idx) * u.packed_stride_mse + pack_idx;
    let packed_mse = K_cold_mse[base_mse];
    let idx = unpack_mse_index(packed_mse, lane);
    let centroid = codebook_centroids[idx];
    let norm = f32(scales_k[key_pos * u.num_kv_heads + kv_head_idx]);
    let mse_val = centroid * norm;

    // QJL residual stage: reconstruct residual from 1-bit signs
    // r_approx[tid] = ||r|| * Σ_j P^T[tid][j] * sign_j / sqrt(d)
    let rnorm = f32(residual_norms_k[key_pos * u.num_kv_heads + kv_head_idx]);
    let res_base = (key_pos * u.num_kv_heads + kv_head_idx) * u.packed_stride_residual;
    var residual: f32 = 0.0;
    let head_dim = u.head_dim;
    for (var j: u32 = 0u; j < head_dim; j++) {
        let word_idx = j / 32u;
        let bit_idx = j % 32u;
        let sign = unpack_sign_bit(residual_k[res_base + word_idx], bit_idx);
        // P^T[tid][j] = P[j][tid], P was scaled by 1/sqrt(d) at generation
        let p_val = qjl_matrix[j * head_dim + tid];
        residual += p_val * sign;
    }
    residual *= rnorm;

    return mse_val + residual * norm;
}

fn dequant_cold_v_prod(key_pos: u32, kv_head_idx: u32, tid: u32) -> f32 {
    let pack_idx = tid / PACK_FACTOR_MSE;
    if (pack_idx >= u.packed_stride_mse) { return 0.0; }
    let lane = tid - pack_idx * PACK_FACTOR_MSE;
    let base_mse = (key_pos * u.num_kv_heads + kv_head_idx) * u.packed_stride_mse + pack_idx;
    let packed_mse = V_cold_mse[base_mse];
    let idx = unpack_mse_index(packed_mse, lane);
    let centroid = codebook_centroids[idx];
    let norm = f32(scales_v[key_pos * u.num_kv_heads + kv_head_idx]);
    let mse_val = centroid * norm;

    let rnorm = f32(residual_norms_v[key_pos * u.num_kv_heads + kv_head_idx]);
    let res_base = (key_pos * u.num_kv_heads + kv_head_idx) * u.packed_stride_residual;
    var residual: f32 = 0.0;
    let head_dim = u.head_dim;
    for (var j: u32 = 0u; j < head_dim; j++) {
        let word_idx = j / 32u;
        let bit_idx = j % 32u;
        let sign = unpack_sign_bit(residual_v[res_base + word_idx], bit_idx);
        let p_val = qjl_matrix[j * head_dim + tid];
        residual += p_val * sign;
    }
    residual *= rnorm;

    return mse_val + residual * norm;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let head_dim = u.head_dim;
    let cold_len = u.cold_len;
    let hot_len = u.hot_len;
    let total_len = cold_len + hot_len;
    let scale = u.scale;
    let kv_head_idx = get_kv_head_idx(head_idx);

    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || head_dim > MAX_HEAD_DIM || total_len > MAX_KV_LEN) {
        return;
    }

    let valid = tid < head_dim;

    var q_val: f32 = 0.0;
    if (valid) {
        q_val = Q[head_idx * head_dim + tid];
    }

    // Rotate Q for cold-tier comparisons
    if (valid) {
        var rot_q: f32 = 0.0;
        let row_base = tid * head_dim;
        for (var j: u32 = 0u; j < head_dim; j++) {
            rot_q += rotation_matrix[row_base + j] * Q[head_idx * head_dim + j];
        }
        shared_rot_q[tid] = rot_q;
    }
    workgroupBarrier();

    if (valid) { shared_acc[tid] = 0.0; }

    // Cold tier scores (rotated domain + residual correction)
    for (var k_pos: u32 = 0u; k_pos < cold_len; k_pos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            k_val = dequant_cold_k_prod(k_pos, kv_head_idx, tid);
        }
        let partial = shared_rot_q[tid] * k_val;
        shared_partial[tid] = select(0.0, partial, valid);
        workgroupBarrier();

        if (tid == 0u) {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) { dot += shared_partial[d]; }
            var s = dot * scale;
            if (u.attn_softcap > 0.0) { s = tanh(s / u.attn_softcap) * u.attn_softcap; }
            if (is_masked(k_pos)) { s = -3.402823e+38; }
            shared_scores[k_pos] = s;
        }
        workgroupBarrier();
    }

    // Hot tier scores
    for (var k_pos: u32 = 0u; k_pos < hot_len; k_pos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            let k_idx = get_hot_pos(k_pos);
            k_val = f32(K_hot[k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid]);
        }
        let partial = q_val * k_val;
        shared_partial[tid] = select(0.0, partial, valid);
        workgroupBarrier();

        if (tid == 0u) {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) { dot += shared_partial[d]; }
            var s = dot * scale;
            if (u.attn_softcap > 0.0) { s = tanh(s / u.attn_softcap) * u.attn_softcap; }
            let abs_key = u.hot_start + k_pos;
            if (is_masked(abs_key)) { s = -3.402823e+38; }
            shared_scores[cold_len + k_pos] = s;
        }
        workgroupBarrier();
    }

    // Softmax
    if (tid == 0u) {
        var m: f32 = -3.402823e+38;
        for (var k: u32 = 0u; k < total_len; k++) { m = max(m, shared_scores[k]); }
        shared_max = m;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var k: u32 = 0u; k < total_len; k++) {
            let w = exp(shared_scores[k] - shared_max);
            shared_scores[k] = w;
            s += w;
        }
        shared_sum = s;
    }
    workgroupBarrier();

    let inv_sum = select(0.0, 1.0 / shared_sum, shared_sum > 0.0);

    // Value accumulation
    if (valid) {
        var acc_cold: f32 = 0.0;
        var acc_hot: f32 = 0.0;

        for (var si: u32 = 0u; si < cold_len; si++) {
            acc_cold += shared_scores[si] * dequant_cold_v_prod(si, kv_head_idx, tid);
        }
        for (var si: u32 = cold_len; si < total_len; si++) {
            let hp = si - cold_len;
            let vi = get_hot_pos(hp);
            acc_hot += shared_scores[si] * f32(V_hot[vi * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid]);
        }

        shared_acc[tid] = acc_cold;
        shared_partial[tid] = acc_hot;
    }
    workgroupBarrier();

    // Inverse rotation for cold V accumulation
    if (valid) {
        var inv_rot: f32 = 0.0;
        for (var j: u32 = 0u; j < head_dim; j++) {
            inv_rot += rotation_matrix[j * head_dim + tid] * shared_acc[j];
        }
        output[head_idx * head_dim + tid] = (inv_rot + shared_partial[tid]) * inv_sum;
    }
}
