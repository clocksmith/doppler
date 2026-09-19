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
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read> softmax: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_scores: array<f32>;

var<workgroup> shared_reduce: array<f32, MAX_WORKGROUP_SIZE>;

fn output_value_dot(query_token: u32, head: u32, key_token: u32) -> f32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    let kv_head = head / heads_per_kv;
    let output_base = (query_token * u.num_heads + head) * u.head_dim;
    let value_base = (key_token * u.num_kv_heads + kv_head) * u.head_dim;
    var result = 0.0;
    for (var dim = 0u; dim < u.head_dim; dim = dim + 1u) {
        result = result + grad_output[output_base + dim] * value[value_base + dim];
    }
    return result;
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
    let row_base = row * u.seq_len;
    var local_dot = 0.0;
    for (var key_token = lane; key_token < u.seq_len; key_token = key_token + WORKGROUP_SIZE) {
        local_dot = local_dot
            + softmax[row_base + key_token]
            * output_value_dot(query_token, head, key_token);
    }
    shared_reduce[lane] = local_dot;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
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
    let row_dot = shared_reduce[0];
    for (var key_token = lane; key_token < u.seq_len; key_token = key_token + WORKGROUP_SIZE) {
        let probability = softmax[row_base + key_token];
        grad_scores[row_base + key_token] = probability
            * (output_value_dot(query_token, head, key_token) - row_dot);
    }
}
