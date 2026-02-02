// Conv2D Kernel (NCHW, f16)

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    in_channels: u32,
    out_channels: u32,
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
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<f16>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_height = u.out_height;
    let out_width = u.out_width;
    let out_size = u.out_channels * out_height * out_width;
    if (idx >= out_size) {
        return;
    }

    let out_spatial = out_height * out_width;
    let out_c = idx / out_spatial;
    let rem = idx - out_c * out_spatial;
    let out_y = rem / out_width;
    let out_x = rem - out_y * out_width;

    var sum: f32 = f32(bias[out_c]);

    let in_channels = u.in_channels;
    let k_h = u.kernel_h;
    let k_w = u.kernel_w;
    let stride = u.stride;
    let pad = i32(u.pad);

    for (var ic: u32 = 0u; ic < in_channels; ic = ic + 1u) {
        for (var ky: u32 = 0u; ky < k_h; ky = ky + 1u) {
            let in_y = i32(out_y * stride + ky) - pad;
            if (in_y < 0 || in_y >= i32(u.height)) {
                continue;
            }
            for (var kx: u32 = 0u; kx < k_w; kx = kx + 1u) {
                let in_x = i32(out_x * stride + kx) - pad;
                if (in_x < 0 || in_x >= i32(u.width)) {
                    continue;
                }
                let input_idx = (ic * u.height + u32(in_y)) * u.width + u32(in_x);
                let weight_idx = (((out_c * in_channels + ic) * k_h + ky) * k_w + kx);
                sum = sum + f32(input[input_idx]) * f32(weight[weight_idx]);
            }
        }
    }

    output[idx] = f16(sum);
}
