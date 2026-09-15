// Input IDs are unique; each invocation owns one logit write.
override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    vocab_size: u32,
    token_count: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.token_count) { return; }
    let token = token_ids[gid.x];
    var negative_infinity_bits = 0xff800000u;
    if (token < u.vocab_size) { logits[token] = bitcast<f32>(negative_infinity_bits); }
}
