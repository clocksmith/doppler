// energy_update.wgsl
// Gradient step on state towards target: state -= stepSize * gradientScale * (state - target).

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    count: u32,
    stepSize: f32,
    gradientScale: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read> targetBuf: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.count) {
        return;
    }
    let diff = state[idx] - targetBuf[idx];
    state[idx] = state[idx] - (u.stepSize * u.gradientScale * diff);
}
