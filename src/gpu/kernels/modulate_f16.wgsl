// Modulate kernel (f16)
// Applies per-channel affine and optional gating.

struct Uniforms {
    num_tokens: u32,
    hidden_size: u32,
    scale_offset: u32,
    shift_offset: u32,
    gate_offset: u32,
    has_gate: u32,
    add_one: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> mod: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_tokens * u.hidden_size;
    if (idx >= total) {
        return;
    }

    let dim = idx % u.hidden_size;
    let raw_scale = f32(mod[u.scale_offset + dim]);
    let shift = f32(mod[u.shift_offset + dim]);
    let scale = select(raw_scale, 1.0 + raw_scale, u.add_one != 0u);
    var value = f32(input[idx]) * scale + shift;
    if (u.has_gate != 0u) {
        let gate = f32(mod[u.gate_offset + dim]);
        value = value * gate;
    }
    output[idx] = f16(value);
}
