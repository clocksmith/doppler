// BDPA Factored Matrix Multiplication (FMM) Decode Kernel (f16)
//
// Implements Basis-Decomposed Paged Attention to drastically reduce VRAM bandwidth.
// Replaces the $O(N \cdot d)$ Key/Value read with:
// 1. $O(N \cdot 3)$ Execution Index Read (I_flat)
// 2. $O(N \cdot 1)$ Quantized Residual Read (P_delta)
// 3. $O(r \cdot d)$ Basis Read (T_basis) where r << N

enable f16;

const MAX_KV_LEN: u32 = 2048u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    seq_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,       // The absolute position of the query token
    attn_softcap: f32,
    sliding_window: u32,
    kv_len_source: u32,
    _pad_1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f16>;
// The 3-Buffer BDPA Cache Hierarchy
@group(0) @binding(2) var<storage, read> T_basis_k: array<f16>;
@group(0) @binding(3) var<storage, read> T_basis_v: array<f16>;
@group(0) @binding(4) var<storage, read> P_delta_k: array<i32>; // Packed Int8 (4 per i32)
@group(0) @binding(5) var<storage, read> P_delta_v: array<i32>; 
@group(0) @binding(6) var<storage, read> I_flat: array<u32>;    // SoA [BasisPtr, DeltaPtr, OrigPos]
@group(0) @binding(7) var<storage, read> rope_cos: array<f32>;  // Needed for on-the-fly RoPE
@group(0) @binding(8) var<storage, read> rope_sin: array<f32>;
@group(0) @binding(9) var<storage, read_write> output: array<f16>;

var<workgroup> shared_scores: array<f32, MAX_KV_LEN>;
var<workgroup> shared_partial: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> shared_acc: array<f32, MAX_WORKGROUP_SIZE>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

// On-the-fly RoPE reconstruction 
fn apply_rope_to_key(raw_k: f32, pos: u32, dim_idx: u32, head_dim: u32, kv_head_idx: u32) -> f32 {
    // A simplified RoPE applied directly to the reconstructed element
    // Full RoPE needs adjacent pair elements [2i, 2i+1] which complicates
    // standard 1D thread layouts, but can be managed by tracking adjacent memory
    
    let rope_offset = pos * head_dim;
    let cos_val = rope_cos[rope_offset + dim_idx];
    let sin_val = rope_sin[rope_offset + dim_idx];
    
    // In actual implementation we need adjacent dimensional state.
    // For this prototype, we assume RoPE already applied or use the standard phase.
    // *TODO: Plumb full Llama/Gemma RoPE pair extraction here*
    return raw_k * cos_val; // simplified stub
}

// Extract Int8 from packed i32 buffer
fn unpack_int8(packed: i32, idx: u32) -> f32 {
    let shift = (idx % 4u) * 8u;
    let byte_val = (packed >> shift) & 0xFF;
    // sign extend
    let signed_val: f32 = f32((byte_val ^ 0x80) - 128);
    let scale_inv = 1.0 / 127.0; // inverse of our JS host scale
    return signed_val * scale_inv;
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

    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || head_dim > WORKGROUP_SIZE || kv_len > MAX_KV_LEN) {
        return;
    }

    let valid = tid < head_dim;

    // Load Query
    var q_val: f32 = 0.0;
    if (valid) {
        let q_offset = head_idx * head_dim + tid;
        q_val = f32(Q[q_offset]);
    }
    if (valid) { shared_acc[tid] = 0.0; }

    // Phase 1: Factored Matmul (Q * K^T)
    for (var i: u32 = 0u; i < kv_len; i++) {
        // Linear scan through Argsorted tokens
        // This memory access is highly sequential for I_flat, 
        // and highly resident in L2 for T_basis
        
        // Structure of Arrays Index lookup
        let basis_id = I_flat[i * 3u + 0u];
        let delta_id = I_flat[i * 3u + 1u]; 
        let orig_pos = I_flat[i * 3u + 2u];
        
        // Causal Masking logic based on Original Position
        let is_masked = (u.is_causal != 0u && orig_pos > u.start_pos);

        var k_val: f32 = 0.0;
        if (valid && !is_masked) {
            // Factor 1: Basis Centroid
            let basis_offset = basis_id * (u.num_kv_heads * head_dim) + (kv_head_idx * head_dim) + tid;
            let basis_k = f32(T_basis_k[basis_offset]);

            // Factor 2: Quantized Residual
            let delta_offset_flat = delta_id * (u.num_kv_heads * head_dim) + (kv_head_idx * head_dim) + tid;
            let delta_packed_idx = delta_offset_flat / 4u;
            let delta_packed = P_delta_k[delta_packed_idx];
            let delta_k = unpack_int8(delta_packed, delta_offset_flat);

            // Reconstruct and RoPE 
            let raw_k = basis_k + delta_k;
            k_val = apply_rope_to_key(raw_k, orig_pos, tid, head_dim, kv_head_idx);
        }

        let partial = q_val * k_val;
        shared_partial[tid] = partial;
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
            if (is_masked) {
                s = -3.402823e+38; 
            }
            shared_scores[i] = s;
        }
        workgroupBarrier();
    }

    // Phase 2: Softmax
    if (tid == 0u) {
        var max_score: f32 = -3.402823e+38;
        for (var k: u32 = 0u; k < kv_len; k++) {
            max_score = max(max_score, shared_scores[k]);
        }
        shared_max = max_score;
    }
    workgroupBarrier();

    let max_score = shared_max;

    if (tid == 0u) {
        var sum_exp: f32 = 0.0;
        for (var k: u32 = 0u; k < kv_len; k++) {
            let w = exp(shared_scores[k] - max_score);
            shared_scores[k] = w;
            sum_exp += w;
        }
        shared_sum = sum_exp;
    }
    workgroupBarrier();

    let sum_exp = shared_sum;
    let inv_sum = select(0.0, 1.0 / sum_exp, sum_exp > 0.0);

    // Phase 3: Multiply by Values (V)
    if (valid) {
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < kv_len; i++) {
            let basis_id = I_flat[i * 3u + 0u];
            let delta_id = I_flat[i * 3u + 1u];
            let orig_pos = I_flat[i * 3u + 2u];
            let is_masked = (u.is_causal != 0u && orig_pos > u.start_pos);

            if (!is_masked) {
                let basis_offset = basis_id * (u.num_kv_heads * head_dim) + (kv_head_idx * head_dim) + tid;
                let basis_v = f32(T_basis_v[basis_offset]);

                let delta_offset_flat = delta_id * (u.num_kv_heads * head_dim) + (kv_head_idx * head_dim) + tid;
                let delta_packed_idx = delta_offset_flat / 4u;
                let delta_packed = P_delta_v[delta_packed_idx];
                let delta_v = unpack_int8(delta_packed, delta_offset_flat);

                let v_val = basis_v + delta_v;
                acc += shared_scores[i] * v_val;
            }
        }
        shared_acc[tid] = acc * inv_sum;
    }
    workgroupBarrier();

    if (valid) {
        let out_offset = head_idx * head_dim + tid;
        output[out_offset] = f16(shared_acc[tid]);
    }
}
