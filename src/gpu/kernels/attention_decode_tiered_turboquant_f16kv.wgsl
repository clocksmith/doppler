// Tiered Decode Attention Kernel (TurboQuant MSE cold, f16 hot, f32 output)
//
// Cold tier: rotation-based codebook dequant with inverse rotation for V.
// Hot tier: direct f16 reads.
// Key insight: ⟨Πq, Πk⟩ = ⟨q, k⟩ — no inverse rotation needed for QK scores.
// V accumulation in rotated domain, then apply Π^T to final output.

enable f16;

const MAX_KV_LEN: u32 = 2048u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_HEAD_DIM: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;
override BIT_WIDTH: u32 = 4u;
override PACK_FACTOR: u32 = 8u;
override NUM_CENTROIDS: u32 = 16u;

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
    packed_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K_hot: array<f16>;
@group(0) @binding(3) var<storage, read> V_hot: array<f16>;
@group(0) @binding(4) var<storage, read> K_cold: array<u32>;
@group(0) @binding(5) var<storage, read> V_cold: array<u32>;
@group(0) @binding(6) var<storage, read> scales_k: array<f16>;
@group(0) @binding(7) var<storage, read> scales_v: array<f16>;
@group(0) @binding(8) var<storage, read_write> output: array<f32>;
@group(0) @binding(9) var<storage, read> rotation_matrix: array<f32>;
@group(0) @binding(10) var<storage, read> codebook_centroids: array<f32>;

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
    if (u.hot_window > 0u) {
        return (u.hot_start + hot_pos) % u.hot_window;
    }
    return hot_pos;
}

fn unpack_index(packed_val: u32, lane: u32) -> u32 {
    let bit_mask = (1u << BIT_WIDTH) - 1u;
    return (packed_val >> (BIT_WIDTH * lane)) & bit_mask;
}

fn dequant_cold_k(key_pos: u32, kv_head_idx: u32, tid: u32) -> f32 {
    let pack_idx = tid / PACK_FACTOR;
    if (pack_idx >= u.packed_stride) { return 0.0; }
    let lane = tid - pack_idx * PACK_FACTOR;
    let base = (key_pos * u.num_kv_heads + kv_head_idx) * u.packed_stride + pack_idx;
    let packed_val = K_cold[base];
    let idx = unpack_index(packed_val, lane);
    let centroid = codebook_centroids[idx];
    let norm = f32(scales_k[key_pos * u.num_kv_heads + kv_head_idx]);
    return centroid * norm;
}

fn dequant_cold_v(key_pos: u32, kv_head_idx: u32, tid: u32) -> f32 {
    let pack_idx = tid / PACK_FACTOR;
    if (pack_idx >= u.packed_stride) { return 0.0; }
    let lane = tid - pack_idx * PACK_FACTOR;
    let base = (key_pos * u.num_kv_heads + kv_head_idx) * u.packed_stride + pack_idx;
    let packed_val = V_cold[base];
    let idx = unpack_index(packed_val, lane);
    let centroid = codebook_centroids[idx];
    let norm = f32(scales_v[key_pos * u.num_kv_heads + kv_head_idx]);
    return centroid * norm;
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

    // Load Q and compute rotated Q for cold-tier dot products
    // ⟨Πq, Πk⟩ = ⟨q, k⟩, so we rotate Q by Π to match rotated-quantized K
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

    if (valid) {
        shared_acc[tid] = 0.0;
    }

    // Cold tier: dot product in rotated domain (dequant gives rotated*norm values)
    for (var k_pos: u32 = 0u; k_pos < cold_len; k_pos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            k_val = dequant_cold_k(k_pos, kv_head_idx, tid);
        }

        // Dot product of rotated Q and dequanted (rotated*norm) K
        let partial = shared_rot_q[tid] * k_val;
        shared_partial[tid] = select(0.0, partial, valid);
        workgroupBarrier();

        if (tid == 0u) {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) {
                dot += shared_partial[d];
            }
            var s = dot * scale;
            if (u.attn_softcap > 0.0) {
                s = tanh(s / u.attn_softcap) * u.attn_softcap;
            }
            if (is_masked(k_pos)) { s = -3.402823e+38; }
            shared_scores[k_pos] = s;
        }
        workgroupBarrier();
    }

    // Hot tier: standard f16 dot product (no rotation needed since stored as f16)
    for (var k_pos: u32 = 0u; k_pos < hot_len; k_pos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            let k_idx = get_hot_pos(k_pos);
            let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            k_val = f32(K_hot[k_offset]);
        }

        let partial = q_val * k_val;
        shared_partial[tid] = select(0.0, partial, valid);
        workgroupBarrier();

        if (tid == 0u) {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) {
                dot += shared_partial[d];
            }
            var s = dot * scale;
            if (u.attn_softcap > 0.0) {
                s = tanh(s / u.attn_softcap) * u.attn_softcap;
            }
            let abs_key = u.hot_start + k_pos;
            if (is_masked(abs_key)) { s = -3.402823e+38; }
            shared_scores[cold_len + k_pos] = s;
        }
        workgroupBarrier();
    }

    // Softmax
    if (tid == 0u) {
        var max_score: f32 = -3.402823e+38;
        for (var k: u32 = 0u; k < total_len; k++) {
            max_score = max(max_score, shared_scores[k]);
        }
        shared_max = max_score;
    }
    workgroupBarrier();

    let max_score = shared_max;
    if (tid == 0u) {
        var sum_exp: f32 = 0.0;
        for (var k: u32 = 0u; k < total_len; k++) {
            let w = exp(shared_scores[k] - max_score);
            shared_scores[k] = w;
            sum_exp += w;
        }
        shared_sum = sum_exp;
    }
    workgroupBarrier();

    let inv_sum = select(0.0, 1.0 / shared_sum, shared_sum > 0.0);

    // Value accumulation
    // Cold V: accumulated in rotated domain, then inverse-rotated
    // Hot V: accumulated directly
    if (valid) {
        var acc_cold_rotated: f32 = 0.0;
        var acc_hot: f32 = 0.0;

        // Cold tier V (in rotated domain)
        for (var score_idx: u32 = 0u; score_idx < cold_len; score_idx++) {
            let v_val = dequant_cold_v(score_idx, kv_head_idx, tid);
            acc_cold_rotated += shared_scores[score_idx] * v_val;
        }

        // Hot tier V (direct f16)
        for (var score_idx: u32 = cold_len; score_idx < total_len; score_idx++) {
            let hot_pos = score_idx - cold_len;
            let v_idx = get_hot_pos(hot_pos);
            let v_offset = v_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            let v_val = f32(V_hot[v_offset]);
            acc_hot += shared_scores[score_idx] * v_val;
        }

        // Store rotated cold accumulation for inverse rotation
        shared_acc[tid] = acc_cold_rotated;
        shared_partial[tid] = acc_hot;
    }
    workgroupBarrier();

    // Apply inverse rotation Π^T to cold V accumulation
    // out[tid] = Σ_j Π^T[tid][j] * acc_cold_rotated[j]  =  Σ_j Π[j][tid] * acc_cold_rotated[j]
    if (valid) {
        var inv_rot_cold: f32 = 0.0;
        for (var j: u32 = 0u; j < head_dim; j++) {
            // Π^T[tid][j] = Π[j][tid]
            inv_rot_cold += rotation_matrix[j * head_dim + tid] * shared_acc[j];
        }

        // Final output: inverse-rotated cold + direct hot, normalized
        let final_val = (inv_rot_cold + shared_partial[tid]) * inv_sum;
        output[head_idx * head_dim + tid] = final_val;
    }
}
