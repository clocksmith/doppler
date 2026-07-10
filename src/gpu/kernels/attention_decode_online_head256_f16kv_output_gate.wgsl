// Online Decode Attention Kernel with output-gate fusion
// (fixed head_dim 256, f32 Q + f16 KV + f32 output)
//
// Extends attention_decode_online_head256_f16kv.wgsl with a f32 output-gate
// buffer. Used only by execution graphs or runtime profiles that pin this
// explicit fused kernel identity.

enable f16;
enable subgroups;

const WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 256u;
const HEAD_DIM: u32 = 256u;
const HEAD_DIM_VECS: u32 = 64u;
override USE_DIRECT_KV_LAYOUT: bool = false;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    seq_len: u32,
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
@group(0) @binding(7) var<storage, read> output_gate: array<f32>;

var<workgroup> shared_q: array<vec4<f32>, HEAD_DIM_VECS>;
var<workgroup> shared_scores: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_v_bases: array<u32, WORKGROUP_SIZE>;
var<workgroup> sg_max: array<f32, MAX_SUBGROUPS>;
var<workgroup> sg_sum: array<f32, MAX_SUBGROUPS>;
var<workgroup> global_max: f32;
var<workgroup> global_sum: f32;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn get_kv_len() -> u32 {
    if (u.kv_len_source == 0u) {
        return u.kv_len;
    }
    return kv_len_buffer[0];
}

fn is_masked(abs_key: u32) -> bool {
    let abs_query = u.start_pos;
    if (u.is_causal != 0u && abs_key > abs_query) { return true; }
    if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
        if (abs_key < abs_query - u.sliding_window + 1u) { return true; }
    }
    return false;
}

fn get_kv_pos(key_pos: u32) -> u32 {
    let abs_key = u.kv_start + key_pos;
    if (USE_DIRECT_KV_LAYOUT) {
        return abs_key;
    }
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

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) sg_tid: u32,
) {
    if (u.head_dim != HEAD_DIM) {
        return;
    }

    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let kv_len = get_kv_len();
    let q_offset = head_idx * HEAD_DIM;

    if (kv_len == 0u) {
        output[q_offset + tid] = 0.0;
        return;
    }

    let subgroup_id = tid / subgroup_size;
    let num_subgroups = (WORKGROUP_SIZE + subgroup_size - 1u) / subgroup_size;
    if (num_subgroups > MAX_SUBGROUPS) {
        return;
    }

    let kv_head_idx = get_kv_head_idx(head_idx);

    if (tid < HEAD_DIM_VECS) {
        let d = tid * 4u;
        shared_q[tid] = vec4<f32>(
            Q[q_offset + d],
            Q[q_offset + d + 1u],
            Q[q_offset + d + 2u],
            Q[q_offset + d + 3u]
        );
    }
    workgroupBarrier();

    var running_max: f32 = -3.402823e+38;
    var running_sum: f32 = 0.0;
    var out_accum: f32 = 0.0;

    var start_k: u32 = 0u;
    if (u.sliding_window > 0u && kv_len > u.sliding_window) {
        start_k = kv_len - u.sliding_window;
        start_k = (start_k / WORKGROUP_SIZE) * WORKGROUP_SIZE;
    }

    for (var k_start: u32 = start_k; k_start < kv_len; k_start = k_start + WORKGROUP_SIZE) {
        let k_pos = k_start + tid;
        let valid_k = k_pos < kv_len;
        var masked = false;
        var score: f32 = -3.402823e+38;

        if (valid_k) {
            let abs_key = u.kv_start + k_pos;
            masked = is_masked(abs_key);
            let k_idx = get_kv_pos(k_pos);
            let k_offset = k_idx * u.num_kv_heads * HEAD_DIM + kv_head_idx * HEAD_DIM;
            if (!USE_DIRECT_KV_LAYOUT) {
                shared_v_bases[tid] = k_offset;
            }
            if (!masked) {
                var dot_accum: f32 = 0.0;
                for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 2u) {
                    let d0 = d4 * 4u;
                    let d1 = d0 + 4u;
                    let k0 = vec4<f32>(
                        f32(K[k_offset + d0]),
                        f32(K[k_offset + d0 + 1u]),
                        f32(K[k_offset + d0 + 2u]),
                        f32(K[k_offset + d0 + 3u])
                    );
                    let k1 = vec4<f32>(
                        f32(K[k_offset + d1]),
                        f32(K[k_offset + d1 + 1u]),
                        f32(K[k_offset + d1 + 2u]),
                        f32(K[k_offset + d1 + 3u])
                    );
                    dot_accum = dot_accum + dot(shared_q[d4], k0) + dot(shared_q[d4 + 1u], k1);
                }
                score = dot_accum * u.scale;
                if (u.attn_softcap > 0.0) {
                    score = tanh(score / u.attn_softcap) * u.attn_softcap;
                }
            }
        }

        let chunk_max = subgroupMax(score);
        if (sg_tid == 0u && subgroup_id < num_subgroups) {
            sg_max[subgroup_id] = chunk_max;
        }
        workgroupBarrier();

        if (tid == 0u) {
            var m: f32 = -3.402823e+38;
            for (var s: u32 = 0u; s < num_subgroups; s++) {
                m = max(m, sg_max[s]);
            }
            global_max = m;
        }
        workgroupBarrier();

        let chunk_max_val = global_max;
        let new_max = max(running_max, chunk_max_val);
        let rescale = exp(running_max - new_max);

        var exp_score: f32 = 0.0;
        if (valid_k && !masked) {
            exp_score = exp(score - new_max);
        }
        shared_scores[tid] = exp_score;

        let chunk_sum = subgroupAdd(exp_score);
        if (sg_tid == 0u && subgroup_id < num_subgroups) {
            sg_sum[subgroup_id] = chunk_sum;
        }
        workgroupBarrier();

        if (tid == 0u) {
            var s: f32 = 0.0;
            for (var i: u32 = 0u; i < num_subgroups; i++) {
                s = s + sg_sum[i];
            }
            global_sum = s;
        }
        workgroupBarrier();

        running_sum = running_sum * rescale + global_sum;
        running_max = new_max;

        out_accum = out_accum * rescale;
        let chunk_len = min(WORKGROUP_SIZE, kv_len - k_start);
        let chunk_len4 = (chunk_len / 4u) * 4u;
        if (USE_DIRECT_KV_LAYOUT) {
            let direct_v_base = (u.kv_start + k_start) * u.num_kv_heads * HEAD_DIM + kv_head_idx * HEAD_DIM;
            let direct_v_stride = u.num_kv_heads * HEAD_DIM;
            for (var k: u32 = 0u; k < chunk_len4; k = k + 4u) {
                let v_base0 = direct_v_base + k * direct_v_stride;
                let v_base1 = direct_v_base + (k + 1u) * direct_v_stride;
                let v_base2 = direct_v_base + (k + 2u) * direct_v_stride;
                let v_base3 = direct_v_base + (k + 3u) * direct_v_stride;
                out_accum = out_accum + shared_scores[k] * f32(V[v_base0 + tid]);
                out_accum = out_accum + shared_scores[k + 1u] * f32(V[v_base1 + tid]);
                out_accum = out_accum + shared_scores[k + 2u] * f32(V[v_base2 + tid]);
                out_accum = out_accum + shared_scores[k + 3u] * f32(V[v_base3 + tid]);
            }
            for (var k: u32 = chunk_len4; k < chunk_len; k = k + 1u) {
                let v_base = direct_v_base + k * direct_v_stride;
                out_accum = out_accum + shared_scores[k] * f32(V[v_base + tid]);
            }
        } else {
            for (var k: u32 = 0u; k < chunk_len4; k = k + 4u) {
                let v_base0 = shared_v_bases[k];
                let v_base1 = shared_v_bases[k + 1u];
                let v_base2 = shared_v_bases[k + 2u];
                let v_base3 = shared_v_bases[k + 3u];
                out_accum = out_accum + shared_scores[k] * f32(V[v_base0 + tid]);
                out_accum = out_accum + shared_scores[k + 1u] * f32(V[v_base1 + tid]);
                out_accum = out_accum + shared_scores[k + 2u] * f32(V[v_base2 + tid]);
                out_accum = out_accum + shared_scores[k + 3u] * f32(V[v_base3 + tid]);
            }
            for (var k: u32 = chunk_len4; k < chunk_len; k = k + 1u) {
                let v_base = shared_v_bases[k];
                out_accum = out_accum + shared_scores[k] * f32(V[v_base + tid]);
            }
        }
        workgroupBarrier();
    }

    let inv_sum = select(0.0, 1.0 / running_sum, running_sum > 0.0);
    output[q_offset + tid] = out_accum * inv_sum * sigmoid(output_gate[q_offset + tid]);
}
