/**
 * MoE Gather Kernel - Gather tokens by expert for batched execution
 *
 * Groups tokens by their selected experts so that each expert can
 * process its assigned tokens in a single batched operation.
 *
 * NOTE: This kernel requires EXPLICIT bind group layout (not 'auto')
 * because count_and_map and gather_tokens use different subsets of bindings.
 *
 * Input:
 *   - hiddenStates [numTokens, hiddenSize]
 *   - indices [numTokens, topK] - selected expert indices per token
 *
 * Output:
 *   - gathered [numExperts, maxTokensPerExpert, hiddenSize]
 *   - tokenCounts [numExperts] - actual token count per expert
 *   - tokenMap [numExperts, maxTokensPerExpert] - original token index mapping
 */

struct MoEGatherUniforms {
    numTokens: u32,          // Number of input tokens
    hiddenSize: u32,         // Hidden dimension
    numExperts: u32,         // Number of experts
    topK: u32,               // Number of experts per token
    maxTokensPerExpert: u32, // Max tokens any expert can receive
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> uniforms: MoEGatherUniforms;
@group(0) @binding(1) var<storage, read> hiddenStates: array<f32>;      // [numTokens, hiddenSize]
@group(0) @binding(2) var<storage, read> expertIndices: array<u32>;     // [numTokens, topK]
@group(0) @binding(3) var<storage, read_write> gathered: array<f32>;    // [numExperts, maxTokensPerExpert, hiddenSize]
@group(0) @binding(4) var<storage, read_write> tokenCounts: array<atomic<u32>>; // [numExperts]
@group(0) @binding(5) var<storage, read_write> tokenMap: array<u32>;    // [numExperts, maxTokensPerExpert, 2] (tokenIdx, kIdx)

// Phase 1: Count tokens per expert and build token map
// Run with numTokens * topK threads
@compute @workgroup_size(256, 1, 1)
fn count_and_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let totalSlots = uniforms.numTokens * uniforms.topK;

    if (tid >= totalSlots) {
        return;
    }

    let tokenIdx = tid / uniforms.topK;
    let kIdx = tid % uniforms.topK;
    let expertIdx = expertIndices[tid];

    // Atomically increment token count for this expert and get slot
    let slot = atomicAdd(&tokenCounts[expertIdx], 1u);

    // Store mapping: which original token goes to this slot
    if (slot < uniforms.maxTokensPerExpert) {
        let mapBase = expertIdx * uniforms.maxTokensPerExpert * 2u + slot * 2u;
        tokenMap[mapBase] = tokenIdx;
        tokenMap[mapBase + 1u] = kIdx;
    }
}

// Phase 2: Gather hidden states based on token map
// Run with numExperts * maxTokensPerExpert * (hiddenSize / 4) threads
@compute @workgroup_size(256, 1, 1)
fn gather_tokens(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let hiddenSize = uniforms.hiddenSize;
    let maxTokensPerExpert = uniforms.maxTokensPerExpert;
    let numExperts = uniforms.numExperts;

    let elementsPerExpert = maxTokensPerExpert * hiddenSize;
    let totalElements = numExperts * elementsPerExpert;

    if (tid >= totalElements) {
        return;
    }

    // Decode position
    let expertIdx = tid / elementsPerExpert;
    let withinExpert = tid % elementsPerExpert;
    let slotIdx = withinExpert / hiddenSize;
    let dimIdx = withinExpert % hiddenSize;

    // Check if this slot is valid (within actual token count)
    let actualCount = atomicLoad(&tokenCounts[expertIdx]);
    if (slotIdx >= actualCount) {
        // Zero out unused slots
        gathered[tid] = 0.0;
        return;
    }

    // Look up original token index from map
    let mapBase = expertIdx * maxTokensPerExpert * 2u + slotIdx * 2u;
    let tokenIdx = tokenMap[mapBase];

    // Gather from original hidden states
    let srcIdx = tokenIdx * hiddenSize + dimIdx;
    gathered[tid] = hiddenStates[srcIdx];
}

// Combined single-pass version for small models
// Each workgroup handles one expert
@compute @workgroup_size(256, 1, 1)
fn gather_single_pass(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let expertIdx = wg_id.x;
    let threadIdx = local_id.x;
    let hiddenSize = uniforms.hiddenSize;
    let numTokens = uniforms.numTokens;
    let topK = uniforms.topK;
    let maxTokensPerExpert = uniforms.maxTokensPerExpert;

    if (expertIdx >= uniforms.numExperts) {
        return;
    }

    // Phase 1: Count tokens for this expert (thread 0 only)
    var tokenCount: u32 = 0u;
    if (threadIdx == 0u) {
        for (var t: u32 = 0u; t < numTokens; t = t + 1u) {
            for (var k: u32 = 0u; k < topK; k = k + 1u) {
                if (expertIndices[t * topK + k] == expertIdx) {
                    if (tokenCount < maxTokensPerExpert) {
                        let mapBase = expertIdx * maxTokensPerExpert * 2u + tokenCount * 2u;
                        tokenMap[mapBase] = t;
                        tokenMap[mapBase + 1u] = k;
                        tokenCount = tokenCount + 1u;
                    }
                }
            }
        }
        atomicStore(&tokenCounts[expertIdx], tokenCount);
    }

    workgroupBarrier();

    // Phase 2: Gather (all threads participate)
    let actualCount = atomicLoad(&tokenCounts[expertIdx]);
    let elementsPerSlot = hiddenSize;
    let totalWork = actualCount * elementsPerSlot;
    let workPerThread = (totalWork + 255u) / 256u;

    for (var i: u32 = 0u; i < workPerThread; i = i + 1u) {
        let workIdx = threadIdx * workPerThread + i;
        if (workIdx >= totalWork) {
            break;
        }

        let slotIdx = workIdx / hiddenSize;
        let dimIdx = workIdx % hiddenSize;

        let mapBase = expertIdx * maxTokensPerExpert * 2u + slotIdx * 2u;
        let tokenIdx = tokenMap[mapBase];

        let srcIdx = tokenIdx * hiddenSize + dimIdx;
        let dstIdx = expertIdx * maxTokensPerExpert * hiddenSize + slotIdx * hiddenSize + dimIdx;

        gathered[dstIdx] = hiddenStates[srcIdx];
    }
}

// Optimized version: Gather with vec4 loads
@compute @workgroup_size(64, 1, 1)
fn gather_tokens_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let hiddenSize = uniforms.hiddenSize;
    let maxTokensPerExpert = uniforms.maxTokensPerExpert;
    let numExperts = uniforms.numExperts;
    let vec4PerToken = hiddenSize / 4u;

    let vec4PerExpert = maxTokensPerExpert * vec4PerToken;
    let totalVec4s = numExperts * vec4PerExpert;

    if (tid >= totalVec4s) {
        return;
    }

    // Decode position
    let expertIdx = tid / vec4PerExpert;
    let withinExpert = tid % vec4PerExpert;
    let slotIdx = withinExpert / vec4PerToken;
    let vec4Idx = withinExpert % vec4PerToken;

    // Check if slot is valid
    let actualCount = atomicLoad(&tokenCounts[expertIdx]);
    let dstBase = tid * 4u;

    if (slotIdx >= actualCount) {
        gathered[dstBase] = 0.0;
        gathered[dstBase + 1u] = 0.0;
        gathered[dstBase + 2u] = 0.0;
        gathered[dstBase + 3u] = 0.0;
        return;
    }

    // Look up original token
    let mapBase = expertIdx * maxTokensPerExpert * 2u + slotIdx * 2u;
    let tokenIdx = tokenMap[mapBase];

    // Gather 4 elements
    let srcBase = tokenIdx * hiddenSize + vec4Idx * 4u;
    gathered[dstBase] = hiddenStates[srcBase];
    gathered[dstBase + 1u] = hiddenStates[srcBase + 1u];
    gathered[dstBase + 2u] = hiddenStates[srcBase + 2u];
    gathered[dstBase + 3u] = hiddenStates[srcBase + 3u];
}
