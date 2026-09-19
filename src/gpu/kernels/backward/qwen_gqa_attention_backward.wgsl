override WORKGROUP_SIZE: u32 = 256u;

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
@group(0) @binding(3) var<storage, read> softmax: array<f32>;
@group(0) @binding(4) var<storage, read> grad_scores: array<f32>;
@group(0) @binding(5) var<storage, read> grad_output: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_query: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_key: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_value: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_elements = u.seq_len * u.num_heads * u.head_dim;
    let kv_elements = u.seq_len * u.num_kv_heads * u.head_dim;
    let element = gid.x;
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    if (element < query_elements) {
        let dim = element % u.head_dim;
        let row = element / u.head_dim;
        let head = row % u.num_heads;
        let query_token = row / u.num_heads;
        let kv_head = head / heads_per_kv;
        var result = 0.0;
        let score_base = (query_token * u.num_heads + head) * u.seq_len;
        for (var key_token = 0u; key_token < u.seq_len; key_token = key_token + 1u) {
            let key_index = (key_token * u.num_kv_heads + kv_head) * u.head_dim + dim;
            result = result + grad_scores[score_base + key_token] * key[key_index] * u.scale;
        }
        grad_query[element] = result;
        return;
    }
    let kv_element = element - query_elements;
    if (kv_element < kv_elements) {
        let dim = kv_element % u.head_dim;
        let row = kv_element / u.head_dim;
        let kv_head = row % u.num_kv_heads;
        let key_token = row / u.num_kv_heads;
        var result = 0.0;
        let first_head = kv_head * heads_per_kv;
        for (var head_offset = 0u; head_offset < heads_per_kv; head_offset = head_offset + 1u) {
            let head = first_head + head_offset;
            for (var query_token = 0u; query_token < u.seq_len; query_token = query_token + 1u) {
                let query_index = (query_token * u.num_heads + head) * u.head_dim + dim;
                let score_index = (query_token * u.num_heads + head) * u.seq_len + key_token;
                result = result + grad_scores[score_index] * query[query_index] * u.scale;
            }
        }
        grad_key[kv_element] = result;
        return;
    }
    let value_element = kv_element - kv_elements;
    if (value_element < kv_elements) {
        let dim = value_element % u.head_dim;
        let row = value_element / u.head_dim;
        let kv_head = row % u.num_kv_heads;
        let key_token = row / u.num_kv_heads;
        var result = 0.0;
        let first_head = kv_head * heads_per_kv;
        for (var head_offset = 0u; head_offset < heads_per_kv; head_offset = head_offset + 1u) {
            let head = first_head + head_offset;
            for (var query_token = 0u; query_token < u.seq_len; query_token = query_token + 1u) {
                let score_index = (query_token * u.num_heads + head) * u.seq_len + key_token;
                let output_index = (query_token * u.num_heads + head) * u.head_dim + dim;
                result = result + softmax[score_index] * grad_output[output_index];
            }
        }
        grad_value[value_element] = result;
    }
}
