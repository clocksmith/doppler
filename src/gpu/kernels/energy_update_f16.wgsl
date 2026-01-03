// energy_update_f16.wgsl
// Gradient step on f16 state towards f16 target.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    count: u32,
    stepSize: f32,
    gradientScale: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> state: array<f16>;
@group(0) @binding(2) var<storage, read> targetBuf: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.count) {
        return;
    }
    let diff = f32(state[idx] - targetBuf[idx]);
    let next = f32(state[idx]) - (u.stepSize * u.gradientScale * diff);
    state[idx] = f16(next);
}
