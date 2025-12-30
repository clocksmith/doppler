// Online Softmax Kernel
//
// Numerically stable softmax using online algorithm:
// 1. Track running max while iterating
// 2. Compute exp(x - max) and sum in same pass
// 3. Normalize by sum
//
// Supports softmax along last dimension (axis=-1).

override WORKGROUP_SIZE: u32 = 256u;

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
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

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
    var local_max: f32 = -3.402823e+38;

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
    let inv_sum = 1.0 / global_sum;

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

// Optimized version for small inner size (<= 256)
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
    var val: f32 = -3.402823e+38;
    if (thread_idx < inner_size) {
        val = input[base_offset + thread_idx] / temperature;
    }

    // Find max
    shared_max[thread_idx] = val;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
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

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = shared_sum[0];

    // Write normalized output
    if (thread_idx < inner_size) {
        output[base_offset + thread_idx] = exp_val / global_sum;
    }
}

// Online softmax - single pass algorithm
// More memory efficient but requires careful implementation
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn softmax_online(
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

    // Online algorithm: track max and sum simultaneously
    var m: f32 = -3.402823e+38;  // Running max
    var d: f32 = 0.0;            // Running sum of exp(x - m)

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let x = input[base_offset + idx] / temperature;

            // Update running max and rescale sum
            let m_new = max(m, x);
            d = d * exp(m - m_new) + exp(x - m_new);
            m = m_new;
        }
    }

    // Store for reduction
    shared_max[thread_idx] = m;
    shared_sum[thread_idx] = d;
    workgroupBarrier();

    // Pairwise reduction combining (max, sum) pairs
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            let m1 = shared_max[thread_idx];
            let d1 = shared_sum[thread_idx];
            let m2 = shared_max[thread_idx + stride];
            let d2 = shared_sum[thread_idx + stride];

            let m_new = max(m1, m2);
            let d_new = d1 * exp(m1 - m_new) + d2 * exp(m2 - m_new);

            shared_max[thread_idx] = m_new;
            shared_sum[thread_idx] = d_new;
        }
        workgroupBarrier();
    }

    let global_max = shared_max[0];
    let global_sum = shared_sum[0];
    let inv_sum = 1.0 / global_sum;

    workgroupBarrier();

    // Write normalized output
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let x = input[base_offset + idx] / temperature;
            output[base_offset + idx] = exp(x - global_max) * inv_sum;
        }
    }
}

// In-place softmax (output = input buffer)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn softmax_inplace(
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

    // Pass 1: Find max
    var local_max: f32 = -3.402823e+38;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            local_max = max(local_max, output[base_offset + idx] / temperature);
        }
    }

    shared_max[thread_idx] = local_max;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_max[thread_idx] = max(shared_max[thread_idx], shared_max[thread_idx + stride]);
        }
        workgroupBarrier();
    }
    let global_max = shared_max[0];

    // Pass 2: exp and sum
    var local_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let exp_val = exp(output[base_offset + idx] / temperature - global_max);
            output[base_offset + idx] = exp_val;  // Store intermediate
            local_sum = local_sum + exp_val;
        }
    }

    shared_sum[thread_idx] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / shared_sum[0];

    workgroupBarrier();

    // Pass 3: Normalize
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            output[base_offset + idx] = output[base_offset + idx] * inv_sum;
        }
    }
}

// Log softmax - useful for cross-entropy loss
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn log_softmax(
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

    // Pass 1: Find max
    var local_max: f32 = -3.402823e+38;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            local_max = max(local_max, input[base_offset + idx] / temperature);
        }
    }

    shared_max[thread_idx] = local_max;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_max[thread_idx] = max(shared_max[thread_idx], shared_max[thread_idx + stride]);
        }
        workgroupBarrier();
    }
    let global_max = shared_max[0];

    // Pass 2: Sum of exp
    var local_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            local_sum = local_sum + exp(input[base_offset + idx] / temperature - global_max);
        }
    }

    shared_sum[thread_idx] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }
    let log_sum = log(shared_sum[0]);

    workgroupBarrier();

    // Write log softmax: log(exp(x - max) / sum) = (x - max) - log(sum)
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let x = input[base_offset + idx] / temperature;
            output[base_offset + idx] = (x - global_max) - log_sum;
        }
    }
}
