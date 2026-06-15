// Softmax Kernels
//
// Numerically stable softmax:
// 1. Reduce max for each row.
// 2. Compute exp(x - max) and reduce sum.
// 3. Normalize by the reduced sum.
//
// Supports softmax along last dimension (axis=-1).
// Subgroup variants use subgroupMax/subgroupAdd for faster reductions.

override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.402823e+38;

struct Uniforms {
    inner_size: u32,    // Size of dimension to softmax over
    outer_size: u32,    // Product of all other dimensions
    temperature: f32,   // Temperature scaling (divide logits by this before softmax)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Shared memory for reduction
var<workgroup> shared_max: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;

// Main softmax kernel - one workgroup per row
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row_idx = wg_id.x;
    let thread_idx = local_id.x;
    let inner_size = u.inner_size;
    let temperature = u.temperature;

    if (row_idx >= u.outer_size) {
        return;
    }

    let base_offset = row_idx * inner_size;
    let elements_per_thread = (inner_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Pass 1: Find maximum (for numerical stability)
    var local_max: f32 = NEG_INF;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let val = input[base_offset + idx] / temperature;
            local_max = max(local_max, val);
        }
    }

    shared_max[thread_idx] = local_max;
    workgroupBarrier();

    // Parallel reduction for max
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_max[thread_idx] = max(shared_max[thread_idx], shared_max[thread_idx + stride]);
        }
        workgroupBarrier();
    }

    let global_max = shared_max[0];

    // Pass 2: Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let val = input[base_offset + idx] / temperature;
            let exp_val = exp(val - global_max);
            local_sum = local_sum + exp_val;
        }
    }

    shared_sum[thread_idx] = local_sum;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = shared_sum[0];
    // Guard against division by zero when all exp values underflow
    let inv_sum = select(0.0, 1.0 / global_sum, global_sum > 0.0);

    workgroupBarrier();

    // Pass 3: Normalize and write output
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let val = input[base_offset + idx] / temperature;
            let exp_val = exp(val - global_max);
            output[base_offset + idx] = exp_val * inv_sum;
        }
    }
}

// Optimized version for small inner size (<= WORKGROUP_SIZE)
// Each thread handles one element
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn softmax_small(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row_idx = wg_id.x;
    let thread_idx = local_id.x;
    let inner_size = u.inner_size;
    let temperature = u.temperature;

    if (row_idx >= u.outer_size) {
        return;
    }

    let base_offset = row_idx * inner_size;

    // Load and scale value
    var val: f32 = NEG_INF;
    if (thread_idx < inner_size) {
        val = input[base_offset + thread_idx] / temperature;
    }

    // Find max
    shared_max[thread_idx] = val;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_max[thread_idx] = max(shared_max[thread_idx], shared_max[thread_idx + stride]);
        }
        workgroupBarrier();
    }

    let global_max = shared_max[0];

    // Compute exp and sum
    var exp_val: f32 = 0.0;
    if (thread_idx < inner_size) {
        exp_val = exp(val - global_max);
    }

    shared_sum[thread_idx] = exp_val;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = shared_sum[0];
    // Guard against division by zero when all exp values underflow
    let inv_sum = select(0.0, 1.0 / global_sum, global_sum > 0.0);

    // Write normalized output
    if (thread_idx < inner_size) {
        output[base_offset + thread_idx] = exp_val * inv_sum;
    }
}
