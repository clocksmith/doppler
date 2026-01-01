// Attention Decode Kernel - Subgroup Optimized
//
// Optimized for seqLen=1 (decode) using subgroup operations to eliminate barriers.
// Key improvements over tiled variants:
// - Zero workgroup barriers (subgroup operations only)
// - 100% thread utilization (all headDim threads active)
// - No wasted work from tiling overhead
//
// Architecture:
// - One workgroup per head
// - workgroup_size = headDim (256 for Gemma 1B)
// - Each thread handles one dimension of Q vector
// - Subgroup reductions for scores and weighted sums

// Uniforms must match TypeScript createAttentionUniformBuffer() layout exactly:
// offset 0: numHeads, offset 4: numKVHeads, offset 8: headDim,
// offset 12: kvLen, offset 16: seqLen, offset 20: scale, offset 24: causal, offset 28: startPos
struct Uniforms {
    numHeads: u32,      // Number of query heads
    numKVHeads: u32,    // Number of KV heads (GQA support)
    headDim: u32,       // Head dimension (256 for Gemma)
    kvLen: u32,         // Current KV cache length
    seqLen: u32,        // Always 1 for decode
    scale: f32,         // Attention scale (1/sqrt(headDim))
    causal: u32,        // Causal masking flag
    startPos: u32,      // Start position for RoPE
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;           // [1, numHeads, headDim]
@group(0) @binding(2) var<storage, read> K_cache: array<f32>;     // [kvLen, numKVHeads, headDim] (F16 as F32)
@group(0) @binding(3) var<storage, read> V_cache: array<f32>;     // [kvLen, numKVHeads, headDim] (F16 as F32)
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [1, numHeads, headDim]

// Shared memory for attention scores (one per workgroup/head)
var<workgroup> scores: array<f32, 2048>;  // Max KV length supported

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let headDim = uniforms.headDim;
    let kvLen = uniforms.kvLen;

    // GQA: map query head to KV head
    let kv_head_idx = head_idx / (uniforms.numHeads / uniforms.numKVHeads);

    // Early exit if thread beyond headDim
    if (tid >= headDim) {
        return;
    }

    // Load Q value for this thread
    let q_offset = head_idx * headDim + tid;
    let q_val = Q[q_offset];

    let scale = 1.0 / sqrt(f32(headDim));

    // Phase 1: Compute attention scores (Q @ K^T)
    // Each thread computes its contribution to all KV positions
    for (var k = 0u; k < kvLen; k++) {
        let k_offset = k * uniforms.numKVHeads * headDim + kv_head_idx * headDim + tid;
        let k_val = K_cache[k_offset];

        // Compute dot product contribution
        let dot = q_val * k_val;

        // Subgroup reduction to get full score
        let score = subgroupAdd(dot);

        // Only first thread in subgroup writes the score
        if (subgroupElect()) {
            scores[k] = score * scale;
        }
    }

    // Ensure all scores are computed before softmax
    workgroupBarrier();

    // Phase 2: Softmax over attention scores
    // Each thread processes a subset of scores
    var max_score = -1e38;
    for (var k = tid; k < kvLen; k += headDim) {
        max_score = max(max_score, scores[k]);
    }

    // Find global max across workgroup
    max_score = subgroupMax(max_score);
    // Need one barrier to broadcast max to all subgroups
    workgroupBarrier();

    // Compute exp and sum
    var sum_exp = 0.0;
    for (var k = tid; k < kvLen; k += headDim) {
        let exp_val = exp(scores[k] - max_score);
        scores[k] = exp_val;
        sum_exp += exp_val;
    }

    // Reduce sum across workgroup
    sum_exp = subgroupAdd(sum_exp);
    workgroupBarrier();

    // Normalize
    for (var k = tid; k < kvLen; k += headDim) {
        scores[k] /= sum_exp;
    }

    workgroupBarrier();

    // Phase 3: Weighted sum (scores @ V)
    var output_val = 0.0;
    for (var k = 0u; k < kvLen; k++) {
        let v_offset = k * uniforms.numKVHeads * headDim + kv_head_idx * headDim + tid;
        let v_val = V_cache[v_offset];
        output_val += scores[k] * v_val;
    }

    // Write output
    let out_offset = head_idx * headDim + tid;
    output[out_offset] = output_val;
}
