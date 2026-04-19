// Contiguous Decode Attention Kernel (TurboQuant MSE, f32 output)
//
// All KV entries are TurboQuant-quantized (no hot/cold split).
// For full-attention models (Qwen 3.5, LFM) that cannot use tiered layout.
// Q is rotated by Π, dot products computed in rotated domain,
// V accumulated in rotated domain then inverse-rotated.

enable f16;

const MAX_KV_LEN: u32 = 2048u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_HEAD_DIM: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;
override BIT_WIDTH: u32 = 4u;
override PACK_FACTOR: u32 = 8u;

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
    packed_stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K_packed: array<u32>;
@group(0) @binding(3) var<storage, read> V_packed: array<u32>;
@group(0) @binding(4) var<storage, read> scales_k: array<f16>;
@group(0) @binding(5) var<storage, read> scales_v: array<f16>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<storage, read> rotation_matrix: array<f32>;
@group(0) @binding(8) var<storage, read> codebook_centroids: array<f32>;

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

fn unpack_index(packed_val: u32, lane: u32) -> u32 {
    return (packed_val >> (BIT_WIDTH * lane)) & ((1u << BIT_WIDTH) - 1u);
}

fn dequant_k(key_pos: u32, kv_head_idx: u32, tid: u32) -> f32 {
    let pack_idx = tid / PACK_FACTOR;
    if (pack_idx >= u.packed_stride) { return 0.0; }
    let lane = tid - pack_idx * PACK_FACTOR;
    let base = (key_pos * u.num_kv_heads + kv_head_idx) * u.packed_stride + pack_idx;
    let idx = unpack_index(K_packed[base], lane);
    let norm = f32(scales_k[key_pos * u.num_kv_heads + kv_head_idx]);
    return codebook_centroids[idx] * norm;
}

fn dequant_v(key_pos: u32, kv_head_idx: u32, tid: u32) -> f32 {
    let pack_idx = tid / PACK_FACTOR;
    if (pack_idx >= u.packed_stride) { return 0.0; }
    let lane = tid - pack_idx * PACK_FACTOR;
    let base = (key_pos * u.num_kv_heads + kv_head_idx) * u.packed_stride + pack_idx;
    let idx = unpack_index(V_packed[base], lane);
    let norm = f32(scales_v[key_pos * u.num_kv_heads + kv_head_idx]);
    return codebook_centroids[idx] * norm;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let head_dim = u.head_dim;
    let kv_len = u.kv_len;
    let scale = u.scale;
    let kv_head_idx = get_kv_head_idx(head_idx);

    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || head_dim > MAX_HEAD_DIM || kv_len > MAX_KV_LEN) {
        return;
    }

    let valid = tid < head_dim;

    // Rotate Q by Π for dot products in rotated domain
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

    // Score computation
    for (var k_pos: u32 = 0u; k_pos < kv_len; k_pos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            k_val = dequant_k(k_pos, kv_head_idx, tid);
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

    // Softmax
    if (tid == 0u) {
        var m: f32 = -3.402823e+38;
        for (var k: u32 = 0u; k < kv_len; k++) { m = max(m, shared_scores[k]); }
        shared_max = m;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var k: u32 = 0u; k < kv_len; k++) {
            let w = exp(shared_scores[k] - shared_max);
            shared_scores[k] = w;
            s += w;
        }
        shared_sum = s;
    }
    workgroupBarrier();

    let inv_sum = select(0.0, 1.0 / shared_sum, shared_sum > 0.0);

    // V accumulation in rotated domain
    if (valid) {
        var acc: f32 = 0.0;
        for (var si: u32 = 0u; si < kv_len; si++) {
            acc += shared_scores[si] * dequant_v(si, kv_head_idx, tid);
        }
        shared_acc[tid] = acc;
    }
    workgroupBarrier();

    // Inverse rotation Π^T and normalize
    if (valid) {
        var inv_rot: f32 = 0.0;
        for (var j: u32 = 0u; j < head_dim; j++) {
            inv_rot += rotation_matrix[j * head_dim + tid] * shared_acc[j];
        }
        output[head_idx * head_dim + tid] = inv_rot * inv_sum;
    }
}
