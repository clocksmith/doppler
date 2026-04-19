// TurboQuant KV Quantization Kernel (MSE-optimal, no outlier)
//
// Applies random rotation Π, then scalar-quantizes each coordinate via
// Max-Lloyd codebook lookup. Stores packed indices + L2 norm as scale.

enable f16;

const MAX_HEAD_DIM: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;
override BIT_WIDTH: u32 = 4u;
override PACK_FACTOR: u32 = 8u;
override NUM_BOUNDARIES: u32 = 15u;

struct Uniforms {
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    num_tokens: u32,
    packed_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input_k: array<f16>;
@group(0) @binding(2) var<storage, read> input_v: array<f16>;
@group(0) @binding(3) var<storage, read_write> output_k: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_v: array<u32>;
@group(0) @binding(5) var<storage, read_write> scales_k: array<f16>;
@group(0) @binding(6) var<storage, read_write> scales_v: array<f16>;
@group(0) @binding(7) var<storage, read> rotation_matrix: array<f32>;
@group(0) @binding(8) var<storage, read> codebook_boundaries: array<f32>;

var<workgroup> shared_k: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_v: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_rotated_k: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_rotated_v: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_qk: array<u32, MAX_HEAD_DIM>;
var<workgroup> shared_qv: array<u32, MAX_HEAD_DIM>;
var<workgroup> shared_norm_k: f32;
var<workgroup> shared_norm_v: f32;

fn quantize_scalar(val: f32, num_boundaries: u32, boundaries_offset: u32) -> u32 {
    var idx: u32 = 0u;
    for (var b: u32 = 0u; b < num_boundaries; b++) {
        if (val > codebook_boundaries[boundaries_offset + b]) {
            idx = b + 1u;
        }
    }
    return idx;
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

    // Step 1: Load K/V into shared memory
    if (valid) {
        shared_k[tid] = f32(input_k[input_base + tid]);
        shared_v[tid] = f32(input_v[input_base + tid]);
    }
    workgroupBarrier();

    // Step 2: Compute L2 norms (thread 0 reduces)
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

    // Step 3: Apply rotation Π and normalize to unit vector
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

    // Step 4: Scalar quantize each rotated coordinate
    if (valid) {
        shared_qk[tid] = quantize_scalar(shared_rotated_k[tid], NUM_BOUNDARIES, 0u);
        shared_qv[tid] = quantize_scalar(shared_rotated_v[tid], NUM_BOUNDARIES, 0u);
    }
    workgroupBarrier();

    // Step 5: Pack low-bit indices into u32
    let out_token = u.start_pos + token_idx;
    let packed_base = (out_token * u.num_kv_heads + head_idx) * u.packed_stride;

    if (valid && (tid % PACK_FACTOR) == 0u) {
        let pack_idx = tid / PACK_FACTOR;
        if (pack_idx < u.packed_stride) {
            var packed_k: u32 = 0u;
            var packed_v: u32 = 0u;
            let bit_mask = (1u << BIT_WIDTH) - 1u;
            for (var i: u32 = 0u; i < PACK_FACTOR; i++) {
                let lane = tid + i;
                if (lane < head_dim) {
                    packed_k |= (shared_qk[lane] & bit_mask) << (BIT_WIDTH * i);
                    packed_v |= (shared_qv[lane] & bit_mask) << (BIT_WIDTH * i);
                }
            }
            output_k[packed_base + pack_idx] = packed_k;
            output_v[packed_base + pack_idx] = packed_v;
        }
    }

    // Step 6: Store L2 norms as scales
    if (tid == 0u) {
        let scale_idx = out_token * u.num_kv_heads + head_idx;
        scales_k[scale_idx] = f16(norm_k);
        scales_v[scale_idx] = f16(norm_v);
    }
}
