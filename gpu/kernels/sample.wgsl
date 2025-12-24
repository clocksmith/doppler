/**
 * GPU-Side Sampling Kernel
 *
 * Performs temperature scaling, top-k selection, softmax, and sampling
 * entirely on GPU. Only reads back the single selected token ID.
 *
 * Reduces readback from 1MB (256K vocab × 4 bytes) to 4 bytes.
 *
 * Algorithm:
 * 1. Temperature scaling: logits = logits / temperature
 * 2. Parallel top-k: Each workgroup finds local top-k, then merge
 * 3. Softmax on top-k candidates
 * 4. Multinomial sampling with provided random value
 */

// Configuration
const WORKGROUP_SIZE: u32 = 256u;
const MAX_TOP_K: u32 = 128u;  // Max top-k supported

struct SampleUniforms {
    vocabSize: u32,
    topK: u32,
    temperature: f32,
    randomValue: f32,  // Pre-generated random [0,1) for sampling
}

@group(0) @binding(0) var<uniform> uniforms: SampleUniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;              // [vocabSize]
@group(0) @binding(2) var<storage, read_write> output: array<u32>;         // [1] - selected token
@group(0) @binding(3) var<storage, read_write> topkIndices: array<u32>;    // [topK] - intermediate
@group(0) @binding(4) var<storage, read_write> topkLogits: array<f32>;     // [topK] - intermediate

// Shared memory for workgroup-level reduction
var<workgroup> shared_values: array<f32, 256>;
var<workgroup> shared_indices: array<u32, 256>;

// Phase 1: Find local max in each workgroup for parallel top-k
// Each thread scans a chunk of vocabulary, keeps local top element
@compute @workgroup_size(256, 1, 1)
fn find_topk_phase1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let threadIdx = lid.x;
    let globalIdx = gid.x;
    let vocabSize = uniforms.vocabSize;
    let temperature = uniforms.temperature;

    // Each thread finds max in its assigned range
    var localMax: f32 = -3.402823e+38;  // -FLT_MAX
    var localMaxIdx: u32 = 0u;

    // Stride through vocabulary
    var idx = globalIdx;
    while (idx < vocabSize) {
        let val = logits[idx] / temperature;
        if (val > localMax) {
            localMax = val;
            localMaxIdx = idx;
        }
        idx = idx + WORKGROUP_SIZE * 256u;  // 256 workgroups assumed
    }

    shared_values[threadIdx] = localMax;
    shared_indices[threadIdx] = localMaxIdx;
    workgroupBarrier();

    // Reduce within workgroup to find workgroup's top value
    var stride = 128u;
    while (stride > 0u) {
        if (threadIdx < stride) {
            if (shared_values[threadIdx + stride] > shared_values[threadIdx]) {
                shared_values[threadIdx] = shared_values[threadIdx + stride];
                shared_indices[threadIdx] = shared_indices[threadIdx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes workgroup result
    if (threadIdx == 0u) {
        topkLogits[wgid.x] = shared_values[0];
        topkIndices[wgid.x] = shared_indices[0];
    }
}

// Phase 2: Merge workgroup results and select final top-k
// Single workgroup sorts and selects top-k from workgroup results
@compute @workgroup_size(256, 1, 1)
fn find_topk_phase2(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let threadIdx = lid.x;
    let topK = uniforms.topK;

    // Load workgroup results into shared memory
    // Assume <= 256 workgroups from phase 1
    if (threadIdx < 256u) {
        shared_values[threadIdx] = topkLogits[threadIdx];
        shared_indices[threadIdx] = topkIndices[threadIdx];
    }
    workgroupBarrier();

    // Thread 0 does partial selection sort for top-k
    if (threadIdx == 0u) {
        for (var k: u32 = 0u; k < topK && k < 256u; k = k + 1u) {
            var maxIdx = k;
            var maxVal = shared_values[k];

            for (var i: u32 = k + 1u; i < 256u; i = i + 1u) {
                if (shared_values[i] > maxVal) {
                    maxVal = shared_values[i];
                    maxIdx = i;
                }
            }

            if (maxIdx != k) {
                let tmpVal = shared_values[k];
                let tmpIdx = shared_indices[k];
                shared_values[k] = shared_values[maxIdx];
                shared_indices[k] = shared_indices[maxIdx];
                shared_values[maxIdx] = tmpVal;
                shared_indices[maxIdx] = tmpIdx;
            }
        }

        // Write sorted top-k back
        for (var k: u32 = 0u; k < topK; k = k + 1u) {
            topkLogits[k] = shared_values[k];
            topkIndices[k] = shared_indices[k];
        }
    }
}

// Phase 3: Softmax on top-k and sample
@compute @workgroup_size(256, 1, 1)
fn softmax_and_sample(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let threadIdx = lid.x;
    let topK = uniforms.topK;
    let randomVal = uniforms.randomValue;

    // Load top-k logits
    if (threadIdx < topK) {
        shared_values[threadIdx] = topkLogits[threadIdx];
        shared_indices[threadIdx] = topkIndices[threadIdx];
    }
    workgroupBarrier();

    // Thread 0 does softmax and sampling
    if (threadIdx == 0u) {
        // Find max for numerical stability
        var maxVal: f32 = shared_values[0];
        for (var i: u32 = 1u; i < topK; i = i + 1u) {
            maxVal = max(maxVal, shared_values[i]);
        }

        // Compute exp and sum
        var expSum: f32 = 0.0;
        for (var i: u32 = 0u; i < topK; i = i + 1u) {
            let expVal = exp(shared_values[i] - maxVal);
            shared_values[i] = expVal;
            expSum = expSum + expVal;
        }

        // Normalize to probabilities and sample
        let invSum = 1.0 / expSum;
        var cumProb: f32 = 0.0;
        var selectedToken: u32 = shared_indices[topK - 1u];  // Default to last

        for (var i: u32 = 0u; i < topK; i = i + 1u) {
            let prob = shared_values[i] * invSum;
            cumProb = cumProb + prob;
            if (cumProb >= randomVal) {
                selectedToken = shared_indices[i];
                break;
            }
        }

        output[0] = selectedToken;
    }
}

// Combined single-pass version for smaller vocabularies (<= 65536)
// Uses hierarchical reduction within single kernel
@compute @workgroup_size(256, 1, 1)
fn sample_single_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let threadIdx = lid.x;
    let vocabSize = uniforms.vocabSize;
    let topK = min(uniforms.topK, MAX_TOP_K);
    let temperature = uniforms.temperature;
    let randomVal = uniforms.randomValue;

    // Phase 1: Find global max
    var localMax: f32 = -3.402823e+38;
    var localMaxIdx: u32 = 0u;

    var idx = gid.x;
    while (idx < vocabSize) {
        let val = logits[idx] / temperature;
        if (val > localMax) {
            localMax = val;
            localMaxIdx = idx;
        }
        idx = idx + numWg.x * WORKGROUP_SIZE;
    }

    shared_values[threadIdx] = localMax;
    shared_indices[threadIdx] = localMaxIdx;
    workgroupBarrier();

    // Reduce to find workgroup max
    var stride = 128u;
    while (stride > 0u) {
        if (threadIdx < stride) {
            if (shared_values[threadIdx + stride] > shared_values[threadIdx]) {
                shared_values[threadIdx] = shared_values[threadIdx + stride];
                shared_indices[threadIdx] = shared_indices[threadIdx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // For single workgroup, thread 0 can do everything
    if (threadIdx == 0u && numWg.x == 1u) {
        // We have top-1, but need top-k
        // For small vocab, just do the full selection
        // This simplified version selects top-1 only (greedy)
        // Full top-k sampling requires multi-pass for large vocab

        output[0] = shared_indices[0];
    }
}

// Greedy argmax for deterministic decoding (temperature=0 equivalent)
@compute @workgroup_size(256, 1, 1)
fn argmax(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) numWg: vec3<u32>
) {
    let threadIdx = lid.x;
    let globalIdx = gid.x;
    let vocabSize = uniforms.vocabSize;

    // Each thread finds max in its chunk
    var localMax: f32 = -3.402823e+38;
    var localMaxIdx: u32 = 0u;

    var idx = globalIdx;
    while (idx < vocabSize) {
        let val = logits[idx];
        if (val > localMax) {
            localMax = val;
            localMaxIdx = idx;
        }
        idx = idx + numWg.x * WORKGROUP_SIZE;
    }

    shared_values[threadIdx] = localMax;
    shared_indices[threadIdx] = localMaxIdx;
    workgroupBarrier();

    // Reduce within workgroup
    var stride = 128u;
    while (stride > 0u) {
        if (threadIdx < stride) {
            if (shared_values[threadIdx + stride] > shared_values[threadIdx]) {
                shared_values[threadIdx] = shared_values[threadIdx + stride];
                shared_indices[threadIdx] = shared_indices[threadIdx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write workgroup result to global memory
    if (threadIdx == 0u) {
        topkLogits[wgid.x] = shared_values[0];
        topkIndices[wgid.x] = shared_indices[0];
    }
}

// Final reduction for argmax across workgroups
@compute @workgroup_size(256, 1, 1)
fn argmax_reduce(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let threadIdx = lid.x;

    // Load workgroup maxes (up to 256)
    shared_values[threadIdx] = topkLogits[threadIdx];
    shared_indices[threadIdx] = topkIndices[threadIdx];
    workgroupBarrier();

    // Reduce
    var stride = 128u;
    while (stride > 0u) {
        if (threadIdx < stride) {
            if (shared_values[threadIdx + stride] > shared_values[threadIdx]) {
                shared_values[threadIdx] = shared_values[threadIdx + stride];
                shared_indices[threadIdx] = shared_indices[threadIdx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (threadIdx == 0u) {
        output[0] = shared_indices[0];
    }
}
