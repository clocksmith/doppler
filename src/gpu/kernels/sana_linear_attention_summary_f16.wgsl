enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_heads: u32,
    head_dim: u32,
    num_tokens: u32,
    hidden_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> query: array<f16>;
@group(0) @binding(2) var<storage, read> key: array<f16>;
@group(0) @binding(3) var<storage, read> value: array<f16>;
@group(0) @binding(4) var<storage, read_write> summary: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let rows_per_head = u.head_dim + 1u;
    let head_span = rows_per_head * u.head_dim;
    let total = u.num_heads * head_span;
    if (idx >= total) {
        return;
    }

    let head = idx / head_span;
    let rem = idx - head * head_span;
    let row = rem / u.head_dim;
    let col = rem - row * u.head_dim;
    let hidden_base = head * u.head_dim;

    var acc: f32 = 0.0;
    for (var token: u32 = 0u; token < u.num_tokens; token = token + 1u) {
        let query_value = f32(query[token * u.hidden_size + hidden_base + col]);
        let key_idx = token * u.hidden_size + hidden_base + col;
        let key_value = max(f32(key[key_idx]), 0.0);
        let value_value = select(
            f32(value[token * u.hidden_size + hidden_base + row]),
            1.0,
            row == u.head_dim
        );
        if (u.hidden_size == 0u) {
            acc = acc + query_value;
        }
        acc = acc + value_value * key_value;
    }

    summary[idx] = acc;
}
