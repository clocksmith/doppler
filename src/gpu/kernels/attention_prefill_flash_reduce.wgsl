// Flash-Attention Reduction Kernel
//
// Merges per-split (partial_acc, m, l) produced by
// attention_prefill_flash_head256_f16kv.wgsl into the final attention output.
//
// Online softmax merge across splits: for each (query, head),
//   m_global = max(m_s)
//   l_global = sum(exp(m_s - m_global) * l_s)
//   output[d] = sum(exp(m_s - m_global) * acc_s[d]) / l_global
//
// Dispatch: (ceil(query_len * num_heads / WORKGROUP_SIZE), HEAD_DIM_VECS, 1)
// Each thread owns one (query_head_flat, d4) output lane.

const HEAD_DIM: u32 = 256u;
const HEAD_DIM_VECS: u32 = 64u;
const BLOCK_SIZE: u32 = 32u;
const WORKGROUP_SIZE: u32 = 64u;
const MAX_KV_SPLITS: u32 = 16u;

struct ReduceUniforms {
    num_heads: u32,
    query_len: u32,
    num_kv_splits: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: ReduceUniforms;
@group(0) @binding(1) var<storage, read> partial_acc: array<f32>;
@group(0) @binding(2) var<storage, read> partial_m: array<f32>;
@group(0) @binding(3) var<storage, read> partial_l: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let qh_flat = global_id.x;
    let d4 = global_id.y;

    let num_heads = u.num_heads;
    let query_len = u.query_len;
    let num_kv_splits = u.num_kv_splits;

    let total_qh = query_len * num_heads;
    if (qh_flat >= total_qh || d4 >= HEAD_DIM_VECS) {
        return;
    }

    let query_pos = qh_flat / num_heads;
    let head_idx = qh_flat % num_heads;
    let query_block_idx = query_pos / BLOCK_SIZE;
    let thread_within_block = query_pos - query_block_idx * BLOCK_SIZE;
    let qh_idx = query_block_idx * num_heads + head_idx;

    // Gather m_s and l_s across splits to find global m and global l.
    var m_global: f32 = -3.402823e+38;
    // Safe cap on splits so compilers accept the bounded loop.
    let splits = min(num_kv_splits, MAX_KV_SPLITS);
    for (var s: u32 = 0u; s < splits; s = s + 1u) {
        let stats_idx = (qh_idx * num_kv_splits + s) * BLOCK_SIZE + thread_within_block;
        let m_s = partial_m[stats_idx];
        if (m_s > m_global) {
            m_global = m_s;
        }
    }

    // Compute global l as weighted sum of per-split l with exp correction.
    var l_global: f32 = 0.0;
    for (var s: u32 = 0u; s < splits; s = s + 1u) {
        let stats_idx = (qh_idx * num_kv_splits + s) * BLOCK_SIZE + thread_within_block;
        let m_s = partial_m[stats_idx];
        let l_s = partial_l[stats_idx];
        l_global = l_global + l_s * exp(m_s - m_global);
    }

    // Merge acc across splits for this d4.
    var merged = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < splits; s = s + 1u) {
        let stats_idx = (qh_idx * num_kv_splits + s) * BLOCK_SIZE + thread_within_block;
        let m_s = partial_m[stats_idx];
        let correction = exp(m_s - m_global);
        let partial_base = (qh_idx * num_kv_splits + s) * BLOCK_SIZE * HEAD_DIM
                          + thread_within_block * HEAD_DIM
                          + d4 * 4u;
        let partial_vec = vec4<f32>(
            partial_acc[partial_base],
            partial_acc[partial_base + 1u],
            partial_acc[partial_base + 2u],
            partial_acc[partial_base + 3u]
        );
        merged = merged + correction * partial_vec;
    }

    let inv_l = select(0.0, 1.0 / l_global, l_global > 0.0);
    let out_offset = query_pos * num_heads * HEAD_DIM + head_idx * HEAD_DIM + d4 * 4u;
    let final_vec = merged * inv_l;
    output[out_offset] = final_vec.x;
    output[out_offset + 1u] = final_vec.y;
    output[out_offset + 2u] = final_vec.z;
    output[out_offset + 3u] = final_vec.w;
}
