struct Uniforms {
    size: u32,
    _pad0: u32,
    scale: f32,
    invScale: f32,
    qmin: f32,
    qmax: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) {
        return;
    }

    let quantized = clamp(round(input[idx] * u.invScale), u.qmin, u.qmax);
    output[idx] = quantized * u.scale;
}
