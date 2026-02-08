// conv2d_backward_weight.wgsl
//
// Computes gradient wrt weights (dW).
// dW [out_channels, in_channels, kernel_h, kernel_w]
// dW[oc, ic, ky, kx] = sum_over_spatial(dY[oc, oy, ox] * X[ic, oy*stride + ky - pad, ox*stride + kx - pad])

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
@group(0) @binding(1) var<storage, read> grad_output: array<f32>; // dY [out_channels, out_height, out_width]
@group(0) @binding(2) var<storage, read> input: array<f32>;       // X [in_channels, height, width]
@group(0) @binding(3) var<storage, read_write> grad_weight: array<f32>; // dW [out_channels, in_channels, kernel_h, kernel_w]

override WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let weight_idx = gid.x;
    let total_weight_elements = u.out_channels * u.in_channels * u.kernel_h * u.kernel_w;
    
    if (weight_idx >= total_weight_elements) {
        return;
    }

    let weight_ic_stride = u.kernel_h * u.kernel_w;
    let weight_oc_stride = u.in_channels * weight_ic_stride;

    let oc = weight_idx / weight_oc_stride;
    let rem_oc = weight_idx % weight_oc_stride;
    let ic = rem_oc / weight_ic_stride;
    let rem_ic = rem_oc % weight_ic_stride;
    let ky = rem_ic / u.kernel_w;
    let kx = rem_ic % u.kernel_w;

    var sum: f32 = 0.0;
    
    let out_c_offset = oc * u.out_height * u.out_width;
    let in_c_offset = ic * u.height * u.width;

    for (var oy: u32 = 0u; oy < u.out_height; oy = oy + 1u) {
        let iy = oy * u.stride + ky;
        if (iy < u.pad || iy >= u.height + u.pad) { continue; }
        let real_iy = iy - u.pad;

        for (var ox: u32 = 0u; ox < u.out_width; ox = ox + 1u) {
            let ix = ox * u.stride + kx;
            if (ix < u.pad || ix >= u.width + u.pad) { continue; }
            let real_ix = ix - u.pad;

            let dy = grad_output[out_c_offset + oy * u.out_width + ox];
            let x = input[in_c_offset + real_iy * u.width + real_ix];
            sum = sum + dy * x;
        }
    }

    grad_weight[weight_idx] = sum;
}
