// Fused Multi-Head Attention Kernel (small tiles, f16 KV)
//
// Same algorithm as attention_small.wgsl but K/V are stored as f16.
// Shared tiles also use f16 to minimize workgroup storage.

enable f16;

const BLOCK_SIZE: u32 = 32u;
const HEAD_TILE: u32 = 32u;
const MAX_HEAD_DIM: u32 = 256u;

struct AttentionUniforms {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    seqLen: u32,
    queryLen: u32,
    scale: f32,
    isCausal: u32,
    startPos: u32,  // Absolute position offset for causal masking
}

@group(0) @binding(0) var<uniform> uniforms: AttentionUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> shared_K: array<f16, BLOCK_SIZE * HEAD_TILE>;
var<workgroup> shared_V: array<f16, BLOCK_SIZE * HEAD_TILE>;

fn getKVHeadIdx(queryHeadIdx: u32) -> u32 {
    let headsPerKV = uniforms.numHeads / uniforms.numKVHeads;
    return queryHeadIdx / headsPerKV;
}

fn isMasked(queryPos: u32, keyPos: u32) -> bool {
    if (uniforms.isCausal == 0u) { return false; }
    // Use absolute position (queryPos + startPos) for correct causal masking during decode
    return keyPos > (queryPos + uniforms.startPos);
}

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let linear = wg_id.x;
    let numHeads = uniforms.numHeads;
    let headIdx = linear % numHeads;
    let queryBlockIdx = linear / numHeads;
    let threadIdx = local_id.x;

    let kvHeadIdx = getKVHeadIdx(headIdx);
    let headDim = uniforms.headDim;
    let seqLen = uniforms.seqLen;
    let queryLen = uniforms.queryLen;
    let scale = uniforms.scale;

    let queryPos = queryBlockIdx * BLOCK_SIZE + threadIdx;
    let validQuery = queryPos < queryLen;

    var q_local: array<f32, 256>;
    var acc: array<f32, 256>;

    if (validQuery) {
        let q_offset = queryPos * numHeads * headDim + headIdx * headDim;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            q_local[d] = Q[q_offset + d];
            acc[d] = 0.0;
        }
    }

    var m_i: f32 = -3.402823e+38;
    var l_i: f32 = 0.0;

    let numKVBlocks = (seqLen + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    let numHeadTiles = (headDim + HEAD_TILE - 1u) / HEAD_TILE;

    for (var kvBlock: u32 = 0u; kvBlock < numKVBlocks; kvBlock = kvBlock + 1u) {
        let kvBlockStart = kvBlock * BLOCK_SIZE;

        var scores: array<f32, 32>;
        for (var kInit: u32 = 0u; kInit < BLOCK_SIZE; kInit = kInit + 1u) {
            scores[kInit] = 0.0;
        }

        for (var ht: u32 = 0u; ht < numHeadTiles; ht = ht + 1u) {
            let d0 = ht * HEAD_TILE;
            let tileLen = min(HEAD_TILE, headDim - d0);

            let keyPosLoad = kvBlockStart + threadIdx;
            if (keyPosLoad < seqLen) {
                let k_offset = keyPosLoad * uniforms.numKVHeads * headDim + kvHeadIdx * headDim + d0;
                for (var td: u32 = 0u; td < tileLen; td = td + 1u) {
                    shared_K[threadIdx * HEAD_TILE + td] = K[k_offset + td];
                }
            } else {
                for (var td: u32 = 0u; td < tileLen; td = td + 1u) {
                    shared_K[threadIdx * HEAD_TILE + td] = f16(0.0);
                }
            }

            workgroupBarrier();

            if (validQuery) {
                for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                    let keyPos = kvBlockStart + k;
                    if (keyPos >= seqLen) { continue; }
                    if (isMasked(queryPos, keyPos)) { continue; }

                    var dot_partial: f32 = 0.0;
                    for (var td: u32 = 0u; td < tileLen; td = td + 1u) {
                        dot_partial = dot_partial + q_local[d0 + td] * f32(shared_K[k * HEAD_TILE + td]);
                    }
                    scores[k] = scores[k] + dot_partial;
                }
            }

            workgroupBarrier();
        }

        var m_new: f32 = m_i;
        if (validQuery) {
            var block_max: f32 = -3.402823e+38;
            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let keyPos = kvBlockStart + k;
                if (keyPos >= seqLen) { continue; }
                if (isMasked(queryPos, keyPos)) { continue; }

                let s = scores[k] * scale;
                scores[k] = s;
                block_max = max(block_max, s);
            }

            m_new = max(m_i, block_max);
            let correction = exp(m_i - m_new);

            l_i = l_i * correction;
            for (var d: u32 = 0u; d < headDim; d = d + 1u) {
                acc[d] = acc[d] * correction;
            }
        }

        // Accumulate V contribution by tiling headDim again.
        // Barriers must be in uniform control flow, so only the math is guarded.
        for (var ht: u32 = 0u; ht < numHeadTiles; ht = ht + 1u) {
            let d0 = ht * HEAD_TILE;
            let tileLen = min(HEAD_TILE, headDim - d0);

            let keyPosLoad = kvBlockStart + threadIdx;
            if (keyPosLoad < seqLen) {
                let v_offset = keyPosLoad * uniforms.numKVHeads * headDim + kvHeadIdx * headDim + d0;
                for (var td: u32 = 0u; td < tileLen; td = td + 1u) {
                    shared_V[threadIdx * HEAD_TILE + td] = V[v_offset + td];
                }
            } else {
                for (var td: u32 = 0u; td < tileLen; td = td + 1u) {
                    shared_V[threadIdx * HEAD_TILE + td] = f16(0.0);
                }
            }

            workgroupBarrier();

            if (validQuery) {
                for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                    let keyPos = kvBlockStart + k;
                    if (keyPos >= seqLen) { continue; }
                    if (isMasked(queryPos, keyPos)) { continue; }

                    let p = exp(scores[k] - m_new);
                    // Only accumulate l_i on first head tile to avoid double counting
                    if (ht == 0u) {
                        l_i = l_i + p;
                    }

                    for (var td: u32 = 0u; td < tileLen; td = td + 1u) {
                        acc[d0 + td] = acc[d0 + td] + p * f32(shared_V[k * HEAD_TILE + td]);
                    }
                }
            }

            workgroupBarrier();
        }

        if (validQuery) {
            m_i = m_new;
        }
    }

    if (validQuery && l_i > 0.0) {
        let out_offset = queryPos * numHeads * headDim + headIdx * headDim;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            output[out_offset + d] = acc[d] / l_i;
        }
    }
}
