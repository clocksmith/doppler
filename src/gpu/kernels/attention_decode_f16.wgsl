// Fused Multi-Head Attention Kernel (decode, f16 QKV + f16 output)
//
// Same as attention_decode_f16kv.wgsl but Q/K/V are f16 and output is f16.

enable f16;

const MAX_HEAD_DIM: u32 = 64u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

override DECODE_WORKGROUP_SIZE: u32 = 256u;

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
@group(0) @binding(1) var<storage, read> Q: array<f16>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;
@group(0) @binding(6) var<storage, read> page_table: array<u32>;

var<workgroup> row_max: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> row_sum: array<f32, MAX_WORKGROUP_SIZE>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn is_masked(query_pos: u32, key_pos: u32) -> bool {
    let abs_query = query_pos + u.start_pos;
    let abs_key = u.kv_start + key_pos;
    if (u.is_causal != 0u && abs_key > abs_query) { return true; }
    if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
        if (abs_key < abs_query - u.sliding_window + 1u) { return true; }
    }
    return false;
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

fn get_kv_len() -> u32 {
    if (u.kv_len_source == 0u) {
        return u.seq_len;
    }
    return kv_len_buffer[0];
}

@compute @workgroup_size(DECODE_WORKGROUP_SIZE, 1, 1)
fn attention_decode(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let head_idx = wg_id.x;
    let thread_idx = local_id.x;

    if (DECODE_WORKGROUP_SIZE > MAX_WORKGROUP_SIZE) {
        return;
    }

    let kv_head_idx = get_kv_head_idx(head_idx);
    let head_dim = u.head_dim;
    if (head_dim > MAX_HEAD_DIM) {
        return;
    }
    let seq_len = get_kv_len();
    let scale = u.scale;

    let keys_per_thread = (seq_len + DECODE_WORKGROUP_SIZE - 1u) / DECODE_WORKGROUP_SIZE;

    var q_local: array<f32, MAX_HEAD_DIM>;
    let q_offset = head_idx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_local[d] = f32(Q[q_offset + d]);
    }

    var local_max: f32 = -3.402823e+38;
    let query_pos = 0u;

    for (var i: u32 = 0u; i < keys_per_thread; i = i + 1u) {
        let key_pos = thread_idx * keys_per_thread + i;
        if (key_pos >= seq_len) { break; }
        if (is_masked(query_pos, key_pos)) { continue; }

        let k_idx = get_kv_pos(key_pos);
        let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            score = score + q_local[d] * f32(K[k_offset + d]);
        }
        score = score * scale;

        if (u.attn_softcap > 0.0) {
            score = tanh(score / u.attn_softcap) * u.attn_softcap;
        }

        local_max = max(local_max, score);
    }

    row_max[thread_idx] = local_max;
    workgroupBarrier();

    for (var stride: u32 = DECODE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride && thread_idx + stride < DECODE_WORKGROUP_SIZE) {
            row_max[thread_idx] = max(row_max[thread_idx], row_max[thread_idx + stride]);
        }
        workgroupBarrier();
    }

    let global_max = row_max[0];

    var local_sum: f32 = 0.0;
    var local_out: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        local_out[d] = 0.0;
    }

    for (var i: u32 = 0u; i < keys_per_thread; i = i + 1u) {
        let key_pos = thread_idx * keys_per_thread + i;
        if (key_pos >= seq_len) { break; }
        if (is_masked(query_pos, key_pos)) { continue; }

        let k_offset = key_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        let v_idx = get_kv_pos(key_pos);
        let v_offset = v_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            score = score + q_local[d] * f32(K[k_offset + d]);
        }
        score = score * scale;

        if (u.attn_softcap > 0.0) {
            score = tanh(score / u.attn_softcap) * u.attn_softcap;
        }

        let exp_score = exp(score - global_max);
        local_sum = local_sum + exp_score;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            local_out[d] = local_out[d] + exp_score * f32(V[v_offset + d]);
        }
    }

    row_sum[thread_idx] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = DECODE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride && thread_idx + stride < DECODE_WORKGROUP_SIZE) {
            row_sum[thread_idx] = row_sum[thread_idx] + row_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = row_sum[0];
    let inv_sum = select(0.0, 1.0 / global_sum, global_sum > 0.0);

    let out_offset = head_idx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        row_sum[thread_idx] = local_out[d];
        workgroupBarrier();
        if (thread_idx == 0u) {
            var sum: f32 = 0.0;
            for (var t: u32 = 0u; t < DECODE_WORKGROUP_SIZE; t = t + 1u) {
                sum = sum + row_sum[t];
            }
            output[out_offset + d] = f16(sum * inv_sum);
        }
        workgroupBarrier();
    }
}
