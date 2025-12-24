/**
 * Top-K Selection Kernel for MoE Routing
 *
 * Selects top-k experts for each token based on router logits.
 * Optimized for small k (typically 2) and small n (typically 8 experts).
 *
 * Input: softmax probabilities [numTokens, numExperts]
 * Output:
 *   - indices [numTokens, topK] (u32)
 *   - weights [numTokens, topK] (f32, renormalized)
 */

struct TopKUniforms {
    numTokens: u32,      // Number of tokens
    numExperts: u32,     // Number of experts (typically 8)
    topK: u32,           // Number of experts to select (typically 2)
    normalize: u32,      // Whether to renormalize weights (1 = yes)
}

@group(0) @binding(0) var<uniform> uniforms: TopKUniforms;
@group(0) @binding(1) var<storage, read> probs: array<f32>;           // [numTokens, numExperts]
@group(0) @binding(2) var<storage, read_write> outIndices: array<u32>; // [numTokens, topK]
@group(0) @binding(3) var<storage, read_write> outWeights: array<f32>; // [numTokens, topK]

// Workgroup shared memory for sorting
// Supports up to 256 experts (covers DeepSeek-V2's 160, Snowflake Arctic's 128, etc.)
var<workgroup> shared_probs: array<f32, 256>;
var<workgroup> shared_indices: array<u32, 256>;

// Main kernel: one workgroup per token
// Workgroup size 256 to support loading up to 256 experts in parallel
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tokenIdx = wg_id.x;
    let threadIdx = local_id.x;
    let numExperts = uniforms.numExperts;
    let topK = uniforms.topK;

    if (tokenIdx >= uniforms.numTokens) {
        return;
    }

    let baseOffset = tokenIdx * numExperts;

    // Load probabilities into shared memory (first numExperts threads)
    if (threadIdx < numExperts) {
        shared_probs[threadIdx] = probs[baseOffset + threadIdx];
        shared_indices[threadIdx] = threadIdx;
    }
    workgroupBarrier();

    // Simple selection sort for top-k (efficient for small k and n)
    // Only thread 0 does the sorting to avoid race conditions
    if (threadIdx == 0u) {
        // Find top-k by partial selection sort
        for (var k: u32 = 0u; k < topK; k = k + 1u) {
            var maxIdx = k;
            var maxVal = shared_probs[k];

            // Find maximum in remaining elements
            for (var i: u32 = k + 1u; i < numExperts; i = i + 1u) {
                if (shared_probs[i] > maxVal) {
                    maxVal = shared_probs[i];
                    maxIdx = i;
                }
            }

            // Swap if needed
            if (maxIdx != k) {
                let tmpProb = shared_probs[k];
                let tmpIdx = shared_indices[k];
                shared_probs[k] = shared_probs[maxIdx];
                shared_indices[k] = shared_indices[maxIdx];
                shared_probs[maxIdx] = tmpProb;
                shared_indices[maxIdx] = tmpIdx;
            }
        }

        // Compute weight sum for normalization
        var weightSum: f32 = 0.0;
        for (var k: u32 = 0u; k < topK; k = k + 1u) {
            weightSum = weightSum + shared_probs[k];
        }

        // Write output indices and weights
        let outBase = tokenIdx * topK;
        let invSum = select(1.0, 1.0 / weightSum, uniforms.normalize == 1u && weightSum > 0.0);

        for (var k: u32 = 0u; k < topK; k = k + 1u) {
            outIndices[outBase + k] = shared_indices[k];
            outWeights[outBase + k] = shared_probs[k] * invSum;
        }
    }
}

// Optimized version for topK=2, numExperts<=8
// Each thread handles one token
@compute @workgroup_size(256, 1, 1)
fn topk_2_small(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tokenIdx = gid.x;

    if (tokenIdx >= uniforms.numTokens) {
        return;
    }

    let numExperts = uniforms.numExperts;
    let baseOffset = tokenIdx * numExperts;

    // Find top 2 in a single pass
    var top1Idx: u32 = 0u;
    var top1Val: f32 = probs[baseOffset];
    var top2Idx: u32 = 1u;
    var top2Val: f32 = probs[baseOffset + 1u];

    // Ensure top1 >= top2
    if (top2Val > top1Val) {
        let tmpIdx = top1Idx;
        let tmpVal = top1Val;
        top1Idx = top2Idx;
        top1Val = top2Val;
        top2Idx = tmpIdx;
        top2Val = tmpVal;
    }

    // Scan remaining experts
    for (var i: u32 = 2u; i < numExperts; i = i + 1u) {
        let val = probs[baseOffset + i];
        if (val > top1Val) {
            top2Idx = top1Idx;
            top2Val = top1Val;
            top1Idx = i;
            top1Val = val;
        } else if (val > top2Val) {
            top2Idx = i;
            top2Val = val;
        }
    }

    // Renormalize weights
    let weightSum = top1Val + top2Val;
    let invSum = select(1.0, 1.0 / weightSum, uniforms.normalize == 1u && weightSum > 0.0);

    // Write output
    let outBase = tokenIdx * 2u;
    outIndices[outBase] = top1Idx;
    outIndices[outBase + 1u] = top2Idx;
    outWeights[outBase] = top1Val * invSum;
    outWeights[outBase + 1u] = top2Val * invSum;
}

// Fused softmax + top-k for efficiency
// Avoids separate softmax kernel call
// Workgroup size 256 to support up to 256 experts
@compute @workgroup_size(256, 1, 1)
fn softmax_topk(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tokenIdx = wg_id.x;
    let threadIdx = local_id.x;
    let numExperts = uniforms.numExperts;
    let topK = uniforms.topK;

    if (tokenIdx >= uniforms.numTokens) {
        return;
    }

    let baseOffset = tokenIdx * numExperts;

    // Load logits and find max (for numerical stability)
    if (threadIdx < numExperts) {
        shared_probs[threadIdx] = probs[baseOffset + threadIdx];
        shared_indices[threadIdx] = threadIdx;
    }
    workgroupBarrier();

    // Thread 0 does softmax + top-k
    if (threadIdx == 0u) {
        // Find max
        var maxVal: f32 = shared_probs[0];
        for (var i: u32 = 1u; i < numExperts; i = i + 1u) {
            maxVal = max(maxVal, shared_probs[i]);
        }

        // Compute exp and sum
        var expSum: f32 = 0.0;
        for (var i: u32 = 0u; i < numExperts; i = i + 1u) {
            let expVal = exp(shared_probs[i] - maxVal);
            shared_probs[i] = expVal;
            expSum = expSum + expVal;
        }

        // Normalize to get probabilities
        let invExpSum = 1.0 / expSum;
        for (var i: u32 = 0u; i < numExperts; i = i + 1u) {
            shared_probs[i] = shared_probs[i] * invExpSum;
        }

        // Partial selection sort for top-k
        for (var k: u32 = 0u; k < topK; k = k + 1u) {
            var maxIdx = k;
            var maxProb = shared_probs[k];

            for (var i: u32 = k + 1u; i < numExperts; i = i + 1u) {
                if (shared_probs[i] > maxProb) {
                    maxProb = shared_probs[i];
                    maxIdx = i;
                }
            }

            if (maxIdx != k) {
                let tmpProb = shared_probs[k];
                let tmpIdx = shared_indices[k];
                shared_probs[k] = shared_probs[maxIdx];
                shared_indices[k] = shared_indices[maxIdx];
                shared_probs[maxIdx] = tmpProb;
                shared_indices[maxIdx] = tmpIdx;
            }
        }

        // Renormalize top-k weights
        var weightSum: f32 = 0.0;
        for (var k: u32 = 0u; k < topK; k = k + 1u) {
            weightSum = weightSum + shared_probs[k];
        }

        let outBase = tokenIdx * topK;
        let invSum = select(1.0, 1.0 / weightSum, uniforms.normalize == 1u && weightSum > 0.0);

        for (var k: u32 = 0u; k < topK; k = k + 1u) {
            outIndices[outBase + k] = shared_indices[k];
            outWeights[outBase + k] = shared_probs[k] * invSum;
        }
    }
}
