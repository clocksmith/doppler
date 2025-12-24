// Online Softmax Kernel
//
// Numerically stable softmax using online algorithm:
// 1. Track running max while iterating
// 2. Compute exp(x - max) and sum in same pass
// 3. Normalize by sum
//
// Supports softmax along last dimension (axis=-1).

const WORKGROUP_SIZE: u32 = 256u;

struct SoftmaxUniforms {
    innerSize: u32,    // Size of dimension to softmax over
    outerSize: u32,    // Product of all other dimensions
    temperature: f32,  // Temperature scaling (divide logits by this before softmax)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SoftmaxUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Shared memory for reduction
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

// Main softmax kernel - one workgroup per row
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let rowIdx = wg_id.x;
    let threadIdx = local_id.x;
    let innerSize = uniforms.innerSize;
    let temperature = uniforms.temperature;

    if (rowIdx >= uniforms.outerSize) {
        return;
    }

    let baseOffset = rowIdx * innerSize;
    let elementsPerThread = (innerSize + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Pass 1: Find maximum (for numerical stability)
    var local_max: f32 = -3.402823e+38;

    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            let val = input[baseOffset + idx] / temperature;
            local_max = max(local_max, val);
        }
    }

    shared_max[threadIdx] = local_max;
    workgroupBarrier();

    // Parallel reduction for max
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            shared_max[threadIdx] = max(shared_max[threadIdx], shared_max[threadIdx + stride]);
        }
        workgroupBarrier();
    }

    let global_max = shared_max[0];

    // Pass 2: Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;

    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            let val = input[baseOffset + idx] / temperature;
            let exp_val = exp(val - global_max);
            local_sum = local_sum + exp_val;
        }
    }

    shared_sum[threadIdx] = local_sum;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            shared_sum[threadIdx] = shared_sum[threadIdx] + shared_sum[threadIdx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = shared_sum[0];
    let inv_sum = 1.0 / global_sum;

    workgroupBarrier();

    // Pass 3: Normalize and write output
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            let val = input[baseOffset + idx] / temperature;
            let exp_val = exp(val - global_max);
            output[baseOffset + idx] = exp_val * inv_sum;
        }
    }
}

// Optimized version for small inner size (<= 256)
// Each thread handles one element
@compute @workgroup_size(256, 1, 1)
fn softmax_small(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let rowIdx = wg_id.x;
    let threadIdx = local_id.x;
    let innerSize = uniforms.innerSize;
    let temperature = uniforms.temperature;

    if (rowIdx >= uniforms.outerSize) {
        return;
    }

    let baseOffset = rowIdx * innerSize;

    // Load and scale value
    var val: f32 = -3.402823e+38;
    if (threadIdx < innerSize) {
        val = input[baseOffset + threadIdx] / temperature;
    }

    // Find max
    shared_max[threadIdx] = val;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            shared_max[threadIdx] = max(shared_max[threadIdx], shared_max[threadIdx + stride]);
        }
        workgroupBarrier();
    }

    let global_max = shared_max[0];

    // Compute exp and sum
    var exp_val: f32 = 0.0;
    if (threadIdx < innerSize) {
        exp_val = exp(val - global_max);
    }

    shared_sum[threadIdx] = exp_val;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            shared_sum[threadIdx] = shared_sum[threadIdx] + shared_sum[threadIdx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = shared_sum[0];

    // Write normalized output
    if (threadIdx < innerSize) {
        output[baseOffset + threadIdx] = exp_val / global_sum;
    }
}

// Online softmax - single pass algorithm
// More memory efficient but requires careful implementation
@compute @workgroup_size(256, 1, 1)
fn softmax_online(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let rowIdx = wg_id.x;
    let threadIdx = local_id.x;
    let innerSize = uniforms.innerSize;
    let temperature = uniforms.temperature;

    if (rowIdx >= uniforms.outerSize) {
        return;
    }

    let baseOffset = rowIdx * innerSize;
    let elementsPerThread = (innerSize + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Online algorithm: track max and sum simultaneously
    var m: f32 = -3.402823e+38;  // Running max
    var d: f32 = 0.0;            // Running sum of exp(x - m)

    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            let x = input[baseOffset + idx] / temperature;

            // Update running max and rescale sum
            let m_new = max(m, x);
            d = d * exp(m - m_new) + exp(x - m_new);
            m = m_new;
        }
    }

    // Store for reduction
    shared_max[threadIdx] = m;
    shared_sum[threadIdx] = d;
    workgroupBarrier();

    // Pairwise reduction combining (max, sum) pairs
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            let m1 = shared_max[threadIdx];
            let d1 = shared_sum[threadIdx];
            let m2 = shared_max[threadIdx + stride];
            let d2 = shared_sum[threadIdx + stride];

            let m_new = max(m1, m2);
            let d_new = d1 * exp(m1 - m_new) + d2 * exp(m2 - m_new);

            shared_max[threadIdx] = m_new;
            shared_sum[threadIdx] = d_new;
        }
        workgroupBarrier();
    }

    let global_max = shared_max[0];
    let global_sum = shared_sum[0];
    let inv_sum = 1.0 / global_sum;

    workgroupBarrier();

    // Write normalized output
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            let x = input[baseOffset + idx] / temperature;
            output[baseOffset + idx] = exp(x - global_max) * inv_sum;
        }
    }
}

// In-place softmax (output = input buffer)
@compute @workgroup_size(256, 1, 1)
fn softmax_inplace(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let rowIdx = wg_id.x;
    let threadIdx = local_id.x;
    let innerSize = uniforms.innerSize;
    let temperature = uniforms.temperature;

    if (rowIdx >= uniforms.outerSize) {
        return;
    }

    let baseOffset = rowIdx * innerSize;
    let elementsPerThread = (innerSize + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Pass 1: Find max
    var local_max: f32 = -3.402823e+38;
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            local_max = max(local_max, output[baseOffset + idx] / temperature);
        }
    }

    shared_max[threadIdx] = local_max;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            shared_max[threadIdx] = max(shared_max[threadIdx], shared_max[threadIdx + stride]);
        }
        workgroupBarrier();
    }
    let global_max = shared_max[0];

    // Pass 2: exp and sum
    var local_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            let exp_val = exp(output[baseOffset + idx] / temperature - global_max);
            output[baseOffset + idx] = exp_val;  // Store intermediate
            local_sum = local_sum + exp_val;
        }
    }

    shared_sum[threadIdx] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            shared_sum[threadIdx] = shared_sum[threadIdx] + shared_sum[threadIdx + stride];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / shared_sum[0];

    workgroupBarrier();

    // Pass 3: Normalize
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            output[baseOffset + idx] = output[baseOffset + idx] * inv_sum;
        }
    }
}

// Log softmax - useful for cross-entropy loss
@compute @workgroup_size(256, 1, 1)
fn log_softmax(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let rowIdx = wg_id.x;
    let threadIdx = local_id.x;
    let innerSize = uniforms.innerSize;
    let temperature = uniforms.temperature;

    if (rowIdx >= uniforms.outerSize) {
        return;
    }

    let baseOffset = rowIdx * innerSize;
    let elementsPerThread = (innerSize + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Pass 1: Find max
    var local_max: f32 = -3.402823e+38;
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            local_max = max(local_max, input[baseOffset + idx] / temperature);
        }
    }

    shared_max[threadIdx] = local_max;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            shared_max[threadIdx] = max(shared_max[threadIdx], shared_max[threadIdx + stride]);
        }
        workgroupBarrier();
    }
    let global_max = shared_max[0];

    // Pass 2: Sum of exp
    var local_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            local_sum = local_sum + exp(input[baseOffset + idx] / temperature - global_max);
        }
    }

    shared_sum[threadIdx] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride) {
            shared_sum[threadIdx] = shared_sum[threadIdx] + shared_sum[threadIdx + stride];
        }
        workgroupBarrier();
    }
    let log_sum = log(shared_sum[0]);

    workgroupBarrier();

    // Write log softmax: log(exp(x - max) / sum) = (x - max) - log(sum)
    for (var i: u32 = 0u; i < elementsPerThread; i = i + 1u) {
        let idx = threadIdx * elementsPerThread + i;
        if (idx < innerSize) {
            let x = input[baseOffset + idx] / temperature;
            output[baseOffset + idx] = (x - global_max) - log_sum;
        }
    }
}
