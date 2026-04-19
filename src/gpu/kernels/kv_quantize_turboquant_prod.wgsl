// TurboQuant KV Quantization Kernel (TURBOQUANTprod)
//
// Two-stage quantization for unbiased inner products:
//   Stage 1: MSE-optimal scalar quantization at (b-1) bits
//   Stage 2: 1-bit QJL transform on the residual
// Total: b bits per coordinate.

enable f16;

const MAX_HEAD_DIM: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;
override BIT_WIDTH_MSE: u32 = 3u;
override PACK_FACTOR_MSE: u32 = 10u;
override NUM_BOUNDARIES_MSE: u32 = 7u;

struct Uniforms {
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    num_tokens: u32,
    packed_stride_mse: u32,
    packed_stride_residual: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input_k: array<f16>;
@group(0) @binding(2) var<storage, read> input_v: array<f16>;
// Stage 1 (MSE) outputs:
@group(0) @binding(3) var<storage, read_write> output_k_mse: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_v_mse: array<u32>;
@group(0) @binding(5) var<storage, read_write> scales_k: array<f16>;
@group(0) @binding(6) var<storage, read_write> scales_v: array<f16>;
// Stage 2 (QJL 1-bit residual) outputs:
@group(0) @binding(7) var<storage, read_write> residual_k: array<u32>;
@group(0) @binding(8) var<storage, read_write> residual_v: array<u32>;
@group(0) @binding(9) var<storage, read_write> residual_norms_k: array<f16>;
@group(0) @binding(10) var<storage, read_write> residual_norms_v: array<f16>;
// Shared data:
@group(0) @binding(11) var<storage, read> rotation_matrix: array<f32>;
@group(0) @binding(12) var<storage, read> codebook_centroids: array<f32>;
@group(0) @binding(13) var<storage, read> codebook_boundaries: array<f32>;
@group(0) @binding(14) var<storage, read> qjl_matrix: array<f32>;

var<workgroup> shared_k: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_v: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_rotated_k: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_rotated_v: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_residual_k: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_residual_v: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_qk: array<u32, MAX_HEAD_DIM>;
var<workgroup> shared_qv: array<u32, MAX_HEAD_DIM>;
var<workgroup> shared_norm_k: f32;
var<workgroup> shared_norm_v: f32;
var<workgroup> shared_rnorm_k: f32;
var<workgroup> shared_rnorm_v: f32;

fn quantize_scalar(val: f32) -> u32 {
    var idx: u32 = 0u;
    for (var b: u32 = 0u; b < NUM_BOUNDARIES_MSE; b++) {
        if (val > codebook_boundaries[b]) {
            idx = b + 1u;
        }
    }
    return idx;
}

fn dequant_centroid(idx: u32) -> f32 {
    return codebook_centroids[idx];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let token_idx = workgroup_id.y;
    let tid = local_id.x;
    let head_dim = u.head_dim;

    if (head_idx >= u.num_kv_heads || token_idx >= u.num_tokens) {
        return;
    }
    if (head_dim > MAX_HEAD_DIM) {
        return;
    }

    let valid = tid < head_dim;
    let token_base = token_idx * u.num_kv_heads + head_idx;
    let input_base = token_base * head_dim;

    // Step 1: Load K/V
    if (valid) {
        shared_k[tid] = f32(input_k[input_base + tid]);
        shared_v[tid] = f32(input_v[input_base + tid]);
    }
    workgroupBarrier();

    // Step 2: Compute L2 norms
    if (tid == 0u) {
        var sum_k: f32 = 0.0;
        var sum_v: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d++) {
            sum_k += shared_k[d] * shared_k[d];
            sum_v += shared_v[d] * shared_v[d];
        }
        shared_norm_k = sqrt(sum_k);
        shared_norm_v = sqrt(sum_v);
    }
    workgroupBarrier();

    let norm_k = shared_norm_k;
    let norm_v = shared_norm_v;
    let inv_norm_k = select(0.0, 1.0 / norm_k, norm_k > 0.0);
    let inv_norm_v = select(0.0, 1.0 / norm_v, norm_v > 0.0);

    // Step 3: Apply rotation Π and normalize
    if (valid) {
        var rot_k: f32 = 0.0;
        var rot_v: f32 = 0.0;
        let row_base = tid * head_dim;
        for (var j: u32 = 0u; j < head_dim; j++) {
            let pi_val = rotation_matrix[row_base + j];
            rot_k += pi_val * shared_k[j] * inv_norm_k;
            rot_v += pi_val * shared_v[j] * inv_norm_v;
        }
        shared_rotated_k[tid] = rot_k;
        shared_rotated_v[tid] = rot_v;
    }
    workgroupBarrier();

    // Step 4: Stage 1 — MSE-optimal quantization at (b-1) bits
    if (valid) {
        let qk_idx = quantize_scalar(shared_rotated_k[tid]);
        let qv_idx = quantize_scalar(shared_rotated_v[tid]);
        shared_qk[tid] = qk_idx;
        shared_qv[tid] = qv_idx;

        // Compute residual: r = rotated - dequant(quant(rotated))
        shared_residual_k[tid] = shared_rotated_k[tid] - dequant_centroid(qk_idx);
        shared_residual_v[tid] = shared_rotated_v[tid] - dequant_centroid(qv_idx);
    }
    workgroupBarrier();

    // Step 5: Pack MSE indices
    let out_token = u.start_pos + token_idx;
    let mse_packed_base = (out_token * u.num_kv_heads + head_idx) * u.packed_stride_mse;

    if (valid && (tid % PACK_FACTOR_MSE) == 0u) {
        let pack_idx = tid / PACK_FACTOR_MSE;
        if (pack_idx < u.packed_stride_mse) {
            var packed_k: u32 = 0u;
            var packed_v: u32 = 0u;
            let bit_mask = (1u << BIT_WIDTH_MSE) - 1u;
            for (var i: u32 = 0u; i < PACK_FACTOR_MSE; i++) {
                let lane = tid + i;
                let qk = select(0u, shared_qk[lane], lane < head_dim);
                let qv = select(0u, shared_qv[lane], lane < head_dim);
                packed_k |= (qk & bit_mask) << (BIT_WIDTH_MSE * i);
                packed_v |= (qv & bit_mask) << (BIT_WIDTH_MSE * i);
            }
            output_k_mse[mse_packed_base + pack_idx] = packed_k;
            output_v_mse[mse_packed_base + pack_idx] = packed_v;
        }
    }

    // Step 6: Stage 2 — 1-bit QJL on residual
    // Compute P * residual and store sign bits
    // Each thread computes one projected coordinate and contributes 1 sign bit
    if (valid) {
        var proj_k: f32 = 0.0;
        var proj_v: f32 = 0.0;
        let row_base = tid * head_dim;
        for (var j: u32 = 0u; j < head_dim; j++) {
            let p_val = qjl_matrix[row_base + j];
            proj_k += p_val * shared_residual_k[j];
            proj_v += p_val * shared_residual_v[j];
        }
        // Store sign bit: 1 = positive, 0 = negative
        shared_qk[tid] = select(0u, 1u, proj_k >= 0.0);
        shared_qv[tid] = select(0u, 1u, proj_v >= 0.0);
    }
    workgroupBarrier();

    // Compute residual norms (thread 0)
    if (tid == 0u) {
        var rnorm_k: f32 = 0.0;
        var rnorm_v: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d++) {
            rnorm_k += shared_residual_k[d] * shared_residual_k[d];
            rnorm_v += shared_residual_v[d] * shared_residual_v[d];
        }
        shared_rnorm_k = sqrt(rnorm_k);
        shared_rnorm_v = sqrt(rnorm_v);
    }
    workgroupBarrier();

    // Pack 1-bit signs into u32 (32 signs per u32)
    let residual_packed_base = (out_token * u.num_kv_heads + head_idx) * u.packed_stride_residual;
    if (tid == 0u) {
        let num_words = u.packed_stride_residual;
        for (var w: u32 = 0u; w < num_words; w++) {
            var packed_k: u32 = 0u;
            var packed_v: u32 = 0u;
            for (var b: u32 = 0u; b < 32u; b++) {
                let d = w * 32u + b;
                if (d < head_dim) {
                    packed_k |= (shared_qk[d] << b);
                    packed_v |= (shared_qv[d] << b);
                }
            }
            residual_k[residual_packed_base + w] = packed_k;
            residual_v[residual_packed_base + w] = packed_v;
        }
    }

    // Store norms
    if (tid == 0u) {
        let scale_idx = out_token * u.num_kv_heads + head_idx;
        scales_k[scale_idx] = f16(norm_k);
        scales_v[scale_idx] = f16(norm_v);
        residual_norms_k[scale_idx] = f16(shared_rnorm_k);
        residual_norms_v[scale_idx] = f16(shared_rnorm_v);
    }
}
