override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_heads: u32,
    head_dim: u32,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> summary: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_tokens * u.hidden_size;
    if (idx >= total) {
        return;
    }

    let token = idx / u.hidden_size;
    let hidden = idx - token * u.hidden_size;
    let head = hidden / u.head_dim;
    let dim = hidden - head * u.head_dim;
    let rows_per_head = u.head_dim + 1u;
    let head_offset = head * rows_per_head * u.head_dim;
    let hidden_base = head * u.head_dim;

    var numerator: f32 = 0.0;
    var denominator: f32 = 0.0;
    for (var i: u32 = 0u; i < u.head_dim; i = i + 1u) {
        let q_value = max(query[token * u.hidden_size + hidden_base + i], 0.0);
        numerator = numerator + summary[head_offset + dim * u.head_dim + i] * q_value;
        denominator = denominator + summary[head_offset + u.head_dim * u.head_dim + i] * q_value;
    }

    output[idx] = numerator / (denominator + u.eps);
}
