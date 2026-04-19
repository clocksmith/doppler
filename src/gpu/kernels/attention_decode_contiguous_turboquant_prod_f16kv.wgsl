// Contiguous Decode Attention Kernel (TURBOQUANTprod, f32 output)
//
// All KV entries use two-stage dequant (MSE + QJL residual).
// For full-attention models with unbiased inner product requirement.

enable f16;

const MAX_KV_LEN: u32 = 2048u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_HEAD_DIM: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;
override BIT_WIDTH_MSE: u32 = 3u;
override PACK_FACTOR_MSE: u32 = 10u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    seq_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,
    attn_softcap: f32,
    sliding_window: u32,
    packed_stride_mse: u32,
    packed_stride_residual: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K_packed_mse: array<u32>;
@group(0) @binding(3) var<storage, read> V_packed_mse: array<u32>;
@group(0) @binding(4) var<storage, read> scales_k: array<f16>;
@group(0) @binding(5) var<storage, read> scales_v: array<f16>;
@group(0) @binding(6) var<storage, read> residual_k: array<u32>;
@group(0) @binding(7) var<storage, read> residual_v: array<u32>;
@group(0) @binding(8) var<storage, read> residual_norms_k: array<f16>;
@group(0) @binding(9) var<storage, read> residual_norms_v: array<f16>;
@group(0) @binding(10) var<storage, read_write> output: array<f32>;
@group(0) @binding(11) var<storage, read> rotation_matrix: array<f32>;
@group(0) @binding(12) var<storage, read> codebook_centroids: array<f32>;
@group(0) @binding(13) var<storage, read> qjl_matrix: array<f32>;

var<workgroup> shared_scores: array<f32, MAX_KV_LEN>;
var<workgroup> shared_partial: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> shared_acc: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_rot_q: array<f32, MAX_HEAD_DIM>;

fn get_kv_head_idx(qhi: u32) -> u32 {
    return qhi / (u.num_heads / u.num_kv_heads);
}

fn is_masked(abs_key: u32) -> bool {
    let aq = u.start_pos;
    if (u.is_causal != 0u && abs_key > aq) { return true; }
    if (u.sliding_window > 0u && aq >= u.sliding_window) {
        if (abs_key < aq - u.sliding_window + 1u) { return true; }
    }
    return false;
}

fn unpack_mse(packed: u32, lane: u32) -> u32 {
    return (packed >> (BIT_WIDTH_MSE * lane)) & ((1u << BIT_WIDTH_MSE) - 1u);
}

fn sign_bit(packed: u32, bit: u32) -> f32 {
    return select(-1.0, 1.0, ((packed >> bit) & 1u) == 1u);
}

fn dequant_k(kp: u32, kvhi: u32, tid: u32) -> f32 {
    let pi = tid / PACK_FACTOR_MSE;
    if (pi >= u.packed_stride_mse) { return 0.0; }
    let lane = tid - pi * PACK_FACTOR_MSE;
    let base = (kp * u.num_kv_heads + kvhi) * u.packed_stride_mse + pi;
    let idx = unpack_mse(K_packed_mse[base], lane);
    let norm = f32(scales_k[kp * u.num_kv_heads + kvhi]);
    let mse_val = codebook_centroids[idx] * norm;

    let rnorm = f32(residual_norms_k[kp * u.num_kv_heads + kvhi]);
    let rb = (kp * u.num_kv_heads + kvhi) * u.packed_stride_residual;
    var res: f32 = 0.0;
    for (var j: u32 = 0u; j < u.head_dim; j++) {
        res += qjl_matrix[j * u.head_dim + tid] * sign_bit(residual_k[rb + j / 32u], j % 32u);
    }
    return mse_val + res * rnorm * norm;
}

fn dequant_v(kp: u32, kvhi: u32, tid: u32) -> f32 {
    let pi = tid / PACK_FACTOR_MSE;
    if (pi >= u.packed_stride_mse) { return 0.0; }
    let lane = tid - pi * PACK_FACTOR_MSE;
    let base = (kp * u.num_kv_heads + kvhi) * u.packed_stride_mse + pi;
    let idx = unpack_mse(V_packed_mse[base], lane);
    let norm = f32(scales_v[kp * u.num_kv_heads + kvhi]);
    let mse_val = codebook_centroids[idx] * norm;

    let rnorm = f32(residual_norms_v[kp * u.num_kv_heads + kvhi]);
    let rb = (kp * u.num_kv_heads + kvhi) * u.packed_stride_residual;
    var res: f32 = 0.0;
    for (var j: u32 = 0u; j < u.head_dim; j++) {
        res += qjl_matrix[j * u.head_dim + tid] * sign_bit(residual_v[rb + j / 32u], j % 32u);
    }
    return mse_val + res * rnorm * norm;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let hd = u.head_dim;
    let kvl = u.kv_len;
    let sc = u.scale;
    let kvhi = get_kv_head_idx(head_idx);

    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || hd > MAX_HEAD_DIM || kvl > MAX_KV_LEN) { return; }

    let valid = tid < hd;

    if (valid) {
        var rq: f32 = 0.0;
        let rb = tid * hd;
        for (var j: u32 = 0u; j < hd; j++) {
            rq += rotation_matrix[rb + j] * Q[head_idx * hd + j];
        }
        shared_rot_q[tid] = rq;
    }
    workgroupBarrier();

    if (valid) { shared_acc[tid] = 0.0; }

    for (var kp: u32 = 0u; kp < kvl; kp++) {
        var kv: f32 = 0.0;
        if (valid) { kv = dequant_k(kp, kvhi, tid); }
        shared_partial[tid] = select(0.0, shared_rot_q[tid] * kv, valid);
        workgroupBarrier();

        if (tid == 0u) {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hd; d++) { dot += shared_partial[d]; }
            var s = dot * sc;
            if (u.attn_softcap > 0.0) { s = tanh(s / u.attn_softcap) * u.attn_softcap; }
            if (is_masked(kp)) { s = -3.402823e+38; }
            shared_scores[kp] = s;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        var m: f32 = -3.402823e+38;
        for (var k: u32 = 0u; k < kvl; k++) { m = max(m, shared_scores[k]); }
        shared_max = m;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var k: u32 = 0u; k < kvl; k++) {
            let w = exp(shared_scores[k] - shared_max);
            shared_scores[k] = w;
            s += w;
        }
        shared_sum = s;
    }
    workgroupBarrier();

    let inv_sum = select(0.0, 1.0 / shared_sum, shared_sum > 0.0);

    if (valid) {
        var acc: f32 = 0.0;
        for (var si: u32 = 0u; si < kvl; si++) {
            acc += shared_scores[si] * dequant_v(si, kvhi, tid);
        }
        shared_acc[tid] = acc;
    }
    workgroupBarrier();

    if (valid) {
        var ir: f32 = 0.0;
        for (var j: u32 = 0u; j < hd; j++) {
            ir += rotation_matrix[j * hd + tid] * shared_acc[j];
        }
        output[head_idx * hd + tid] = ir * inv_sum;
    }
}
