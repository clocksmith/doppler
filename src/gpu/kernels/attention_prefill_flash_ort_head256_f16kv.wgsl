// Single-Pass Flash Attention — head_dim=256, f16 KV, GQA-aware
//
// Adapted from Microsoft ONNX Runtime's flash_attention.wgsl.template:
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/
//   webgpu/bert/flash_attention.wgsl.template (MIT License)
//
// Key design vs Doppler's attention_prefill_flash_head256_f16kv.wgsl:
// This is SINGLE-PASS (no KV-split + reduce). Each thread owns one query
// position's full output across all K — private q_tile + private o_tile,
// shared K_tile + V_tile for the current K step. Online softmax updates
// m_prev / l_prev / o_ratio per K step without external reduction.
//
// Dispatch: workgroups = (num_heads, ceil(query_len / WORKGROUP_SIZE), 1).
// At Gemma 4 E2B prefill=64 with WORKGROUP_SIZE=64 → 8 × 1 = 8 WGs total,
// each WG doing all flash-attention work for one (head, query-tile).
// Compare to the split+reduce variant at ~32 WGs + reduce pass.
//
// Shapes assumed:
//   head_dim = 256, head_size_vec = 64 (vec4<f16>)
//   max_k_step = 16 (K tokens per load)
//   Shared: shared_K[16][64] + shared_V[16][64] = 16 KB (vec4<f16>)
//   Private: q_tile[64] vec4<f16> + o_tile[64] vec4<f32> ≈ 1.5 KB per thread.
//
// Binding contract matches attention_head256_f16kv.wgsl so the existing
// dispatcher (executeSmallAttention / attention.js) can select this kernel
// without a new bind-group layout.

enable f16;

const BLOCK_SIZE: u32 = 64u;
const WORKGROUP_SIZE: u32 = BLOCK_SIZE;
const HEAD_DIM: u32 = 256u;
const HEAD_DIM_VECS: u32 = 64u;  // head_dim / 4
const MAX_K_STEP: u32 = 16u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    query_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,
    attn_softcap: f32,
    sliding_window: u32,
    kv_len_source: u32,
    kv_start: u32,
    page_size: u32,
    kv_layout: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;
@group(0) @binding(6) var<storage, read> page_table: array<u32>;

var<workgroup> shared_K: array<vec4<f16>, MAX_K_STEP * HEAD_DIM_VECS>;
var<workgroup> shared_V: array<vec4<f16>, MAX_K_STEP * HEAD_DIM_VECS>;

fn zero_vec4_f16() -> vec4<f16> {
    return vec4<f16>(f16(0.0), f16(0.0), f16(0.0), f16(0.0));
}

fn zero_vec4_f32() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn get_kv_len() -> u32 {
    if (u.kv_len_source == 0u) {
        return u.seq_len;
    }
    return kv_len_buffer[0];
}

fn get_kv_pos(key_pos: u32) -> u32 {
    let abs_key = u.kv_start + key_pos;
    if (u.kv_layout == 1u && u.sliding_window > 0u) {
        return abs_key % u.sliding_window;
    }
    if (u.kv_layout == 2u) {
        let page_idx = abs_key / u.page_size;
        let in_page = abs_key - (page_idx * u.page_size);
        let phys_page = page_table[page_idx];
        return phys_page * u.page_size + in_page;
    }
    return abs_key;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if (u.head_dim != HEAD_DIM) { return; }

    let head_idx = wg_id.x;
    let seq_tile = wg_id.y;
    let q_idx_local = local_id.x;
    let q_idx_global = seq_tile * WORKGROUP_SIZE + q_idx_local;
    let kv_head_idx = get_kv_head_idx(head_idx);
    let valid_q = q_idx_global < u.query_len;

    let scale = u.scale;
    let kv_len = get_kv_len();
    // Causal bound for this query: K positions [0, kv_start + q_idx_global].
    // Non-causal: full kv_len.
    let abs_q_pos = u.kv_start + q_idx_global;
    let causal_bound = select(kv_len, abs_q_pos + 1u, u.is_causal != 0u);
    let loop_bound = min(kv_len, causal_bound);

    // ===== Stage 1: Load Q into private memory (one query per thread) =====
    // Q layout: [query_len, num_heads, head_dim] (row-major BSNH single-batch).
    var q_tile: array<vec4<f16>, HEAD_DIM_VECS>;
    if (valid_q) {
        let q_row_offset = q_idx_global * u.num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        for (var i: u32 = 0u; i < HEAD_DIM_VECS; i = i + 1u) {
            let base = q_row_offset + i * 4u;
            q_tile[i] = vec4<f16>(
                f16(Q[base + 0u] * scale),
                f16(Q[base + 1u] * scale),
                f16(Q[base + 2u] * scale),
                f16(Q[base + 3u] * scale)
            );
        }
    } else {
        for (var i: u32 = 0u; i < HEAD_DIM_VECS; i = i + 1u) {
            q_tile[i] = zero_vec4_f16();
        }
    }

    // ===== Online-softmax state per query =====
    var o_tile: array<vec4<f32>, HEAD_DIM_VECS>;
    for (var i: u32 = 0u; i < HEAD_DIM_VECS; i = i + 1u) {
        o_tile[i] = zero_vec4_f32();
    }
    var m_prev: f32 = -3.4028234663852886e+38;  // -inf in f32
    var l_prev: f32 = 0.0;

    // ===== Stage 2: iterate K in chunks of MAX_K_STEP =====
    for (var k_start: u32 = 0u; k_start < loop_bound; k_start = k_start + MAX_K_STEP) {
        let k_step = min(MAX_K_STEP, loop_bound - k_start);

        // ----- Cooperative load of K and V tiles into shared memory -----
        // Load MAX_K_STEP × HEAD_DIM_VECS vec4<f16> = 16 × 64 = 1024 elements.
        // 64 threads × 16 elements each.
        let kv_head_offset = kv_head_idx * u.seq_len * HEAD_DIM;
        for (var idx: u32 = q_idx_local; idx < MAX_K_STEP * HEAD_DIM_VECS; idx = idx + WORKGROUP_SIZE) {
            let k_slot = idx / HEAD_DIM_VECS;
            let vec_idx = idx % HEAD_DIM_VECS;
            if (k_slot < k_step) {
                let kv_pos = get_kv_pos(k_start + k_slot);
                let src_base = kv_head_offset + kv_pos * HEAD_DIM + vec_idx * 4u;
                shared_K[idx] = vec4<f16>(
                    K[src_base + 0u], K[src_base + 1u],
                    K[src_base + 2u], K[src_base + 3u]
                );
                shared_V[idx] = vec4<f16>(
                    V[src_base + 0u], V[src_base + 1u],
                    V[src_base + 2u], V[src_base + 3u]
                );
            } else {
                shared_K[idx] = zero_vec4_f16();
                shared_V[idx] = zero_vec4_f16();
            }
        }
        workgroupBarrier();

        // ----- Compute QK^T for this thread's query vs k_step K positions -----
        // qk[k] = dot(q, shared_K[k * HEAD_DIM_VECS + 0..HEAD_DIM_VECS-1])
        var qk: array<f32, MAX_K_STEP>;
        for (var k: u32 = 0u; k < MAX_K_STEP; k = k + 1u) {
            qk[k] = 0.0;
        }
        for (var i: u32 = 0u; i < HEAD_DIM_VECS; i = i + 1u) {
            let q_v = q_tile[i];
            for (var k: u32 = 0u; k < MAX_K_STEP; k = k + 1u) {
                let k_v = shared_K[k * HEAD_DIM_VECS + i];
                qk[k] = qk[k] + dot(vec4<f32>(q_v), vec4<f32>(k_v));
            }
        }

        // ----- Apply causal / OOB mask -----
        // Mask K positions >= causal_bound to -inf so they contribute 0 to softmax.
        if (valid_q) {
            for (var k: u32 = 0u; k < MAX_K_STEP; k = k + 1u) {
                let abs_k_pos = k_start + k;
                if (abs_k_pos >= loop_bound) {
                    qk[k] = -3.4028234663852886e+38;
                }
            }
        }

        // ----- Online softmax update: new_max, exp(qk - new_max), denom, o_ratio -----
        var local_max: f32 = qk[0];
        for (var k: u32 = 1u; k < MAX_K_STEP; k = k + 1u) {
            local_max = max(local_max, qk[k]);
        }
        let new_max = max(m_prev, local_max);
        var qk_exp: array<f32, MAX_K_STEP>;
        var local_sum: f32 = 0.0;
        for (var k: u32 = 0u; k < MAX_K_STEP; k = k + 1u) {
            qk_exp[k] = exp(qk[k] - new_max);
            local_sum = local_sum + qk_exp[k];
        }
        let prev_scale = exp(m_prev - new_max);
        let l_new = l_prev * prev_scale + local_sum;
        let safe_l = select(l_new, 1.0e-7, l_new == 0.0);

        // Normalise qk_exp and compute output rescale factor.
        for (var k: u32 = 0u; k < MAX_K_STEP; k = k + 1u) {
            qk_exp[k] = qk_exp[k] / safe_l;
        }
        let o_ratio = (l_prev * prev_scale) / safe_l;
        m_prev = new_max;
        l_prev = safe_l;

        // ----- Accumulate O += normalised_attention * V -----
        for (var i: u32 = 0u; i < HEAD_DIM_VECS; i = i + 1u) {
            var acc = o_tile[i] * o_ratio;
            for (var k: u32 = 0u; k < MAX_K_STEP; k = k + 1u) {
                let v_v = vec4<f32>(shared_V[k * HEAD_DIM_VECS + i]);
                acc = acc + v_v * qk_exp[k];
            }
            o_tile[i] = acc;
        }
        workgroupBarrier();
    }

    // ===== Stage 3: Write O =====
    if (valid_q) {
        let out_row_offset = q_idx_global * u.num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        for (var i: u32 = 0u; i < HEAD_DIM_VECS; i = i + 1u) {
            let base = out_row_offset + i * 4u;
            let v = o_tile[i];
            output[base + 0u] = v.x;
            output[base + 1u] = v.y;
            output[base + 2u] = v.z;
            output[base + 3u] = v.w;
        }
    }
}
