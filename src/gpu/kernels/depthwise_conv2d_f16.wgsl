// Depthwise Conv2D Kernel (NCHW, f16)

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    channels: u32,
    height: u32,
    width: u32,
    out_height: u32,
    out_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    pad: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<f16>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_spatial = u.out_height * u.out_width;
    let out_size = u.channels * out_spatial;
    if (idx >= out_size) {
        return;
    }

    let channel = idx / out_spatial;
    let rem = idx - channel * out_spatial;
    let out_y = rem / u.out_width;
    let out_x = rem - out_y * u.out_width;

    var sum: f32 = f32(bias[channel]);
    let pad = i32(u.pad);

    for (var ky: u32 = 0u; ky < u.kernel_h; ky = ky + 1u) {
        let in_y = i32(out_y * u.stride + ky) - pad;
        if (in_y < 0 || in_y >= i32(u.height)) {
            continue;
        }
        for (var kx: u32 = 0u; kx < u.kernel_w; kx = kx + 1u) {
            let in_x = i32(out_x * u.stride + kx) - pad;
            if (in_x < 0 || in_x >= i32(u.width)) {
                continue;
            }
            let input_idx = (channel * u.height + u32(in_y)) * u.width + u32(in_x);
            let weight_idx = ((channel * u.kernel_h + ky) * u.kernel_w + kx);
            sum = sum + f32(input[input_idx]) * f32(weight[weight_idx]);
        }
    }

    output[idx] = f16(sum);
}
