override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    scale: f32,
    causal: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> key: array<f32>;
@group(0) @binding(3) var<storage, read_write> softmax: array<f32>;

var<workgroup> shared_reduce: array<f32, MAX_WORKGROUP_SIZE>;

fn score(query_token: u32, head: u32, key_token: u32) -> f32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    let kv_head = head / heads_per_kv;
    let query_base = (query_token * u.num_heads + head) * u.head_dim;
    let key_base = (key_token * u.num_kv_heads + kv_head) * u.head_dim;
    var result = 0.0;
    for (var dim = 0u; dim < u.head_dim; dim = dim + 1u) {
        result = result + query[query_base + dim] * key[key_base + dim];
    }
    return result * u.scale;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.x;
    let lane = lid.x;
    let query_token = row / u.num_heads;
    let head = row % u.num_heads;
    if (query_token >= u.seq_len) {
        return;
    }
    var local_max = -3.402823466e+38;
    for (var key_token = lane; key_token < u.seq_len; key_token = key_token + WORKGROUP_SIZE) {
        if (u.causal != 0u && key_token > query_token) {
            continue;
        }
        local_max = max(local_max, score(query_token, head, key_token));
    }
    shared_reduce[lane] = local_max;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            shared_reduce[lane] = max(shared_reduce[lane], shared_reduce[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = shared_reduce[0];
    var local_sum = 0.0;
    for (var key_token = lane; key_token < u.seq_len; key_token = key_token + WORKGROUP_SIZE) {
        if (u.causal != 0u && key_token > query_token) {
            continue;
        }
        local_sum = local_sum + exp(score(query_token, head, key_token) - row_max);
    }
    shared_reduce[lane] = local_sum;
    workgroupBarrier();
    stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            shared_reduce[lane] = shared_reduce[lane] + shared_reduce[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inverse_sum = 1.0 / shared_reduce[0];
    let row_base = row * u.seq_len;
    for (var key_token = lane; key_token < u.seq_len; key_token = key_token + WORKGROUP_SIZE) {
        softmax[row_base + key_token] = select(
            exp(score(query_token, head, key_token) - row_max) * inverse_sum,
            0.0,
            u.causal != 0u && key_token > query_token
        );
    }
}
