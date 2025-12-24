// Fused Multi-Head Attention Kernel
//
// Implements fused Q @ K^T → scale → mask → softmax → @ V
// Uses tiled/blocked approach to avoid materializing full attention matrix.
// Supports grouped query attention (GQA) where numKVHeads < numHeads.
//
// Based on Flash Attention principles adapted for WebGPU.

// Tile sizes for blocked attention
const BLOCK_SIZE: u32 = 64u;  // Sequence tile size
const HEAD_TILE: u32 = 64u;   // Head dimension tile

struct AttentionUniforms {
    numHeads: u32,       // Number of query heads
    numKVHeads: u32,     // Number of KV heads (for GQA)
    headDim: u32,        // Dimension per head
    seqLen: u32,         // Current sequence length (for KV)
    queryLen: u32,       // Query length (1 for decode, seqLen for prefill)
    scale: f32,          // 1/sqrt(headDim)
    isCausal: u32,       // Apply causal mask (1 = yes)
    startPos: u32,  // Absolute position offset for causal masking
}

@group(0) @binding(0) var<uniform> uniforms: AttentionUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;       // [queryLen, numHeads, headDim]
@group(0) @binding(2) var<storage, read> K: array<f32>;       // [seqLen, numKVHeads, headDim]
@group(0) @binding(3) var<storage, read> V: array<f32>;       // [seqLen, numKVHeads, headDim]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [queryLen, numHeads, headDim]

// Shared memory for tiled computation
var<workgroup> shared_K: array<f32, 4096>;  // BLOCK_SIZE * HEAD_TILE
var<workgroup> shared_V: array<f32, 4096>;  // BLOCK_SIZE * HEAD_TILE
var<workgroup> shared_scores: array<f32, 4096>;  // BLOCK_SIZE * BLOCK_SIZE

// Online softmax accumulators (per-thread)
// Sized for 256 to support attention_decode workgroup size (prefill uses 64)
var<workgroup> row_max: array<f32, 256>;
var<workgroup> row_sum: array<f32, 256>;

// Get KV head index for grouped query attention
fn getKVHeadIdx(queryHeadIdx: u32) -> u32 {
    // GQA: multiple query heads share one KV head
    let headsPerKV = uniforms.numHeads / uniforms.numKVHeads;
    return queryHeadIdx / headsPerKV;
}

// Check if position should be masked (causal attention)
fn isMasked(queryPos: u32, keyPos: u32) -> bool {
    if (uniforms.isCausal == 0u) {
        return false;
    }
    // For causal attention, query can only attend to keys at same or earlier positions
    return keyPos > (queryPos + uniforms.startPos);
}

// Main attention kernel - one workgroup per (query_block, head)
// Workgroups are dispatched linearly as: numQueryBlocks * numHeads.
// headIdx and queryBlockIdx are derived from workgroup_id.x.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let linear = wg_id.x;
    let headIdx = linear % uniforms.numHeads;
    let queryBlockIdx = linear / uniforms.numHeads;
    let threadIdx = local_id.x;

    let kvHeadIdx = getKVHeadIdx(headIdx);
    let headDim = uniforms.headDim;
    let seqLen = uniforms.seqLen;
    let queryLen = uniforms.queryLen;
    let scale = uniforms.scale;

    // Query position this thread handles
    let queryPos = queryBlockIdx * BLOCK_SIZE + threadIdx;
    let validQuery = queryPos < queryLen;

    // Initialize online softmax accumulators
    var m_i: f32 = -3.402823e+38;  // -inf for max tracking
    var l_i: f32 = 0.0;            // Sum of exp(x - max)
    var acc: array<f32, 64>;       // Accumulator for output [headDim], assuming headDim <= 64

    // Initialize accumulator
    for (var d: u32 = 0u; d < headDim; d = d + 1u) {
        acc[d] = 0.0;
    }

    // Load query for this thread into registers
    var q_local: array<f32, 64>;
    if (validQuery) {
        let q_offset = queryPos * uniforms.numHeads * headDim + headIdx * headDim;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            q_local[d] = Q[q_offset + d];
        }
    }

    // Process key-value blocks
    let numKVBlocks = (seqLen + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    for (var kvBlock: u32 = 0u; kvBlock < numKVBlocks; kvBlock = kvBlock + 1u) {
        let kvBlockStart = kvBlock * BLOCK_SIZE;

        // Collaborative load of K block into shared memory
        let kLoadIdx = kvBlockStart + threadIdx;
        if (kLoadIdx < seqLen) {
            let k_offset = kLoadIdx * uniforms.numKVHeads * headDim + kvHeadIdx * headDim;
            for (var d: u32 = 0u; d < headDim; d = d + 1u) {
                shared_K[threadIdx * headDim + d] = K[k_offset + d];
            }
        } else {
            for (var d: u32 = 0u; d < headDim; d = d + 1u) {
                shared_K[threadIdx * headDim + d] = 0.0;
            }
        }

        // Load V block
        let vLoadIdx = kvBlockStart + threadIdx;
        if (vLoadIdx < seqLen) {
            let v_offset = vLoadIdx * uniforms.numKVHeads * headDim + kvHeadIdx * headDim;
            for (var d: u32 = 0u; d < headDim; d = d + 1u) {
                shared_V[threadIdx * headDim + d] = V[v_offset + d];
            }
        } else {
            for (var d: u32 = 0u; d < headDim; d = d + 1u) {
                shared_V[threadIdx * headDim + d] = 0.0;
            }
        }

        workgroupBarrier();

        // Compute attention scores for this block
        if (validQuery) {
            // Find max in this block (for numerical stability)
            var block_max: f32 = -3.402823e+38;

            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let keyPos = kvBlockStart + k;
                if (keyPos >= seqLen) { continue; }

                // Check causal mask
                if (isMasked(queryPos, keyPos)) { continue; }

                // Compute Q @ K^T for this position
                var score: f32 = 0.0;
                for (var d: u32 = 0u; d < headDim; d = d + 1u) {
                    score = score + q_local[d] * shared_K[k * headDim + d];
                }
                score = score * scale;

                block_max = max(block_max, score);
                shared_scores[threadIdx * BLOCK_SIZE + k] = score;
            }

            // Online softmax update
            let m_new = max(m_i, block_max);
            let correction = exp(m_i - m_new);

            // Rescale previous accumulator
            l_i = l_i * correction;
            for (var d: u32 = 0u; d < headDim; d = d + 1u) {
                acc[d] = acc[d] * correction;
            }

            // Add contribution from this block
            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let keyPos = kvBlockStart + k;
                if (keyPos >= seqLen) { continue; }
                if (isMasked(queryPos, keyPos)) { continue; }

                let score = shared_scores[threadIdx * BLOCK_SIZE + k];
                let p = exp(score - m_new);
                l_i = l_i + p;

                // Accumulate V contribution
                for (var d: u32 = 0u; d < headDim; d = d + 1u) {
                    acc[d] = acc[d] + p * shared_V[k * headDim + d];
                }
            }

            m_i = m_new;
        }

        workgroupBarrier();
    }

    // Normalize by sum and write output
    if (validQuery && l_i > 0.0) {
        let out_offset = queryPos * uniforms.numHeads * headDim + headIdx * headDim;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            output[out_offset + d] = acc[d] / l_i;
        }
    }
}

// Simplified single-query attention for decode step
// More efficient when queryLen == 1
@compute @workgroup_size(256, 1, 1)
fn attention_decode(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let headIdx = wg_id.x;
    let threadIdx = local_id.x;

    let kvHeadIdx = getKVHeadIdx(headIdx);
    let headDim = uniforms.headDim;
    let seqLen = uniforms.seqLen;
    let scale = uniforms.scale;

    // Each thread handles a subset of key positions
    let keysPerThread = (seqLen + 255u) / 256u;

    // Load query (single position)
    var q_local: array<f32, 128>;  // Support up to 128 headDim
    let q_offset = headIdx * headDim;
    for (var d: u32 = 0u; d < headDim; d = d + 1u) {
        q_local[d] = Q[q_offset + d];
    }

    // Compute partial attention scores and find local max
    var local_max: f32 = -3.402823e+38;
    var local_scores: array<f32, 32>;  // Store scores for this thread's keys
    var local_count: u32 = 0u;

    for (var i: u32 = 0u; i < keysPerThread; i = i + 1u) {
        let keyPos = threadIdx * keysPerThread + i;
        if (keyPos >= seqLen) { break; }

        // Causal: can attend to all previous positions (query is at end)
        let k_offset = keyPos * uniforms.numKVHeads * headDim + kvHeadIdx * headDim;

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            score = score + q_local[d] * K[k_offset + d];
        }
        score = score * scale;

        local_scores[i] = score;
        local_max = max(local_max, score);
        local_count = local_count + 1u;
    }

    // Store local max for reduction
    row_max[threadIdx] = local_max;
    workgroupBarrier();

    // Parallel reduction to find global max
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride && threadIdx + stride < 256u) {
            row_max[threadIdx] = max(row_max[threadIdx], row_max[threadIdx + stride]);
        }
        workgroupBarrier();
    }

    let global_max = row_max[0];

    // Compute exp(score - max) and local sum
    var local_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < local_count; i = i + 1u) {
        local_scores[i] = exp(local_scores[i] - global_max);
        local_sum = local_sum + local_scores[i];
    }

    row_sum[threadIdx] = local_sum;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (threadIdx < stride && threadIdx + stride < 256u) {
            row_sum[threadIdx] = row_sum[threadIdx] + row_sum[threadIdx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = row_sum[0];

    // Compute weighted V contribution
    var local_out: array<f32, 128>;
    for (var d: u32 = 0u; d < headDim; d = d + 1u) {
        local_out[d] = 0.0;
    }

    for (var i: u32 = 0u; i < local_count; i = i + 1u) {
        let keyPos = threadIdx * keysPerThread + i;
        let v_offset = keyPos * uniforms.numKVHeads * headDim + kvHeadIdx * headDim;
        let weight = local_scores[i] / global_sum;

        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            local_out[d] = local_out[d] + weight * V[v_offset + d];
        }
    }

    // Reduction for output (atomic add or shared memory reduction)
    // For simplicity, use shared memory
    for (var d: u32 = 0u; d < headDim; d = d + 1u) {
        shared_V[threadIdx * headDim + d] = local_out[d];
    }
    workgroupBarrier();

    // Thread 0 sums all contributions
    if (threadIdx == 0u) {
        let out_offset = headIdx * headDim;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            var sum: f32 = 0.0;
            for (var t: u32 = 0u; t < 256u; t = t + 1u) {
                sum = sum + shared_V[t * headDim + d];
            }
            output[out_offset + d] = sum;
        }
    }
}
