// groupnorm_backward.wgsl
//
// GroupNorm Backward Kernel (GPU)
// Computes dInput per element. 
// dWeight/dBias require cross-spatial reduction (left for future or handled via separate reduce).

struct Uniforms {
    channels: u32,
    height: u32,
    width: u32,
    num_groups: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;        // x
@group(0) @binding(2) var<storage, read> weight: array<f32>;       // gamma (per channel)
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;  // dY
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>; // dX

override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_sum_sq: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_dot_w: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_dot_w_norm: array<f32, MAX_WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let group_idx = wg_id.x;
    let thread_idx = local_id.x;
    
    if (group_idx >= u.num_groups) {
        return;
    }

    let channels_per_group = u.channels / u.num_groups;
    let spatial = u.height * u.width;
    let elements_per_group = channels_per_group * spatial;
    
    let group_c_start = group_idx * channels_per_group;
    
    // -------------------------------------------------------------------------
    // Phase 1: Recompute Mean and Variance for the group
    // -------------------------------------------------------------------------
    var local_sum: f32 = 0.0;
    var local_sum_sq: f32 = 0.0;
    
    for (var i = thread_idx; i < elements_per_group; i += WORKGROUP_SIZE) {
        let c_in_group = i / spatial;
        let s_idx = i % spatial;
        let c = group_c_start + c_in_group;
        let idx = (c * u.height + (s_idx / u.width)) * u.width + (s_idx % u.width);
        // Simplified mapping if NCHW is contiguous spatial
        let real_idx = c * spatial + s_idx; 
        
        let x = input[real_idx];
        local_sum += x;
        local_sum_sq += x * x;
    }

    shared_sum[thread_idx] = local_sum;
    shared_sum_sq[thread_idx] = local_sum_sq;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] += shared_sum[thread_idx + stride];
            shared_sum_sq[thread_idx] += shared_sum_sq[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let n = f32(elements_per_group);
    let mean = shared_sum[0] / n;
    let mean_sq = shared_sum_sq[0] / n;
    let variance = max(0.0, mean_sq - mean * mean);
    let inv_std = 1.0 / sqrt(variance + u.eps);

    workgroupBarrier();

    // -------------------------------------------------------------------------
    // Phase 2: Compute Gradient Stats for the group
    // -------------------------------------------------------------------------
    var local_dot_w: f32 = 0.0;
    var local_dot_w_norm: f32 = 0.0;

    for (var i = thread_idx; i < elements_per_group; i += WORKGROUP_SIZE) {
        let c_in_group = i / spatial;
        let s_idx = i % spatial;
        let c = group_c_start + c_in_group;
        let real_idx = c * spatial + s_idx;
        
        let x = input[real_idx];
        let dy = grad_output[real_idx];
        let w = weight[c];
        let dy_w = dy * w;
        
        local_dot_w += dy_w;
        local_dot_w_norm += dy_w * (x - mean);
    }

    shared_dot_w[thread_idx] = local_dot_w;
    shared_dot_w_norm[thread_idx] = local_dot_w_norm;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (thread_idx < stride) {
            shared_dot_w[thread_idx] += shared_dot_w[thread_idx + stride];
            shared_dot_w_norm[thread_idx] += shared_dot_w_norm[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let sum_dy_w = shared_dot_w[0];
    let sum_dy_w_norm = shared_dot_w_norm[0];
    let term1 = sum_dy_w;
    let term2 = sum_dy_w_norm * (inv_std * inv_std);

    // -------------------------------------------------------------------------
    // Phase 3: Write Gradient Input
    // -------------------------------------------------------------------------
    for (var i = thread_idx; i < elements_per_group; i += WORKGROUP_SIZE) {
        let c_in_group = i / spatial;
        let s_idx = i % spatial;
        let c = group_c_start + c_in_group;
        let real_idx = c * spatial + s_idx;
        
        let x = input[real_idx];
        let dy = grad_output[real_idx];
        let w = weight[c];

        let dx = inv_std * (dy * w - (term1 + (x - mean) * term2) / n);
        grad_input[real_idx] = dx;
    }
}
