// layernorm_backward.wgsl
//
// LayerNorm Backward Kernel (GPU)
// Computes dInput, dWeight, and dBias.
//
// Gradients:
// dInput = (1/N) * inv_std * (
//   N * dY * weight -
//   sum(dY * weight) -
//   (x - mean) * inv_std^2 * sum(dY * weight * (x - mean))
// )
// dWeight = sum_over_batch(dY * norm)
// dBias = sum_over_batch(dY)
//
// Note: This kernel currently computes dInput per token. 
// dWeight and dBias accumulation across batch requires a separate reduction 
// or atomic adds. For now, we focus on dInput as that's the main path 
// for backprop to previous layers. dWeight/dBias support can be added via 
// atomic accumulation or a secondary reduction kernel.
//
// In this simplified version, we output dInput. 
// dWeight/dBias are left for a separate reduce kernel or future expansion.

// =============================================================================
// Override Constants
// =============================================================================

override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
override PARAMS_IS_F16: bool = false; // Weight/bias packed as f16 pairs (unused for now as we read f32)

// =============================================================================
// Uniforms
// =============================================================================

struct Uniforms {
    size: u32,       // hidden_size
    num_tokens: u32, // batch_size * seq_len
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;        // x
@group(0) @binding(2) var<storage, read> weight: array<f32>;       // gamma
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;  // dY
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>; // dX
@group(0) @binding(5) var<storage, read_write> grad_weight: array<atomic<u32>>; // dWeight (sum dY * norm)
@group(0) @binding(6) var<storage, read_write> grad_bias: array<atomic<u32>>;   // dBias (sum dY)

// =============================================================================
// Shared Memory
// =============================================================================

// We need multiple reductions:
// 1. mean (from forward, recomputed here for memory savings)
// 2. mean_sq (for variance)
// 3. sum(dY * weight)
// 4. sum(dY * weight * (x - mean))

var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_sum_sq: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_dot_w: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_dot_w_norm: array<f32, MAX_WORKGROUP_SIZE>;

// Atomic float add helper using bitcast
fn atomicAddFloat(address: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old = atomicLoad(address);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old) + val);
        let res = atomicCompareExchangeWeak(address, old, new_val);
        if (res.exchanged) {
            break;
        }
        old = res.old_value;
    }
}

// =============================================================================
// Entry Point
// =============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // -------------------------------------------------------------------------
    // Phase 1: Recompute Mean and Variance (Forward Pass logic)
    // -------------------------------------------------------------------------
    var local_sum: f32 = 0.0;
    var local_sum_sq: f32 = 0.0;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            local_sum += x;
            local_sum_sq += x * x;
        }
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

    let mean = shared_sum[0] / f32(size);
    let mean_sq = shared_sum_sq[0] / f32(size);
    let variance = max(0.0, mean_sq - mean * mean);
    let inv_std = 1.0 / sqrt(variance + u.eps);

    workgroupBarrier();

    // -------------------------------------------------------------------------
    // Phase 2: Compute Gradients Statistics
    // term1 = sum(dY * weight)
    // term2 = sum(dY * weight * (x - mean))
    // -------------------------------------------------------------------------
    var local_dot_w: f32 = 0.0;
    var local_dot_w_norm: f32 = 0.0;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            let dy = grad_output[base_offset + idx];
            let w = weight[idx];
            let dy_w = dy * w;
            
            local_dot_w += dy_w;
            local_dot_w_norm += dy_w * (x - mean);
            
            // Phase 4 part: accumulate dWeight and dBias
            // norm = (x - mean) * inv_std
            // dWeight[idx] += dY * norm
            // dBias[idx] += dY
            let norm = (x - mean) * inv_std;
            atomicAddFloat(&grad_weight[idx], dy * norm);
            atomicAddFloat(&grad_bias[idx], dy);
        }
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
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            let dy = grad_output[base_offset + idx];
            let w = weight[idx];

            let dx = inv_std * (dy * w - (term1 + (x - mean) * term2) / f32(size));
            grad_input[base_offset + idx] = dx;
        }
    }
}
