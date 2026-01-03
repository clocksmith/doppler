// Upsample2D Kernel (nearest, NCHW, f16)

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    scale: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_spatial = u.out_height * u.out_width;
    let total = u.channels * out_spatial;
    if (idx >= total) {
        return;
    }

    let channel = idx / out_spatial;
    let rem = idx - channel * out_spatial;
    let out_y = rem / u.out_width;
    let out_x = rem - out_y * u.out_width;
    let in_y = out_y / u.scale;
    let in_x = out_x / u.scale;
    let in_idx = (channel * u.in_height + in_y) * u.in_width + in_x;
    output[idx] = input[in_idx];
}
