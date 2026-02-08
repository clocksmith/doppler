// conv2d_backward_input.wgsl
//
// Computes gradient wrt input (dX).
// This is essentially a transposed convolution (or deconvolution).
// For each input pixel [c, y, x], it sums the contributions from all output pixels 
// that it contributed to during the forward pass.

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
@group(0) @binding(2) var<storage, read> weight: array<f32>;      // W [out_channels, in_channels, kernel_h, kernel_w]
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>; // dX [in_channels, height, width]

override WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_idx = gid.x;
    let total_in_elements = u.in_channels * u.height * u.width;
    
    if (in_idx >= total_in_elements) {
        return;
    }

    let in_spatial = u.height * u.width;
    let ic = in_idx / in_spatial;
    let rem = in_idx % in_spatial;
    let iy = rem / u.width;
    let ix = rem % u.width;

    var sum: f32 = 0.0;

    // Weight is [OC, IC, KH, KW]
    let weight_ic_stride = u.kernel_h * u.kernel_w;
    let weight_oc_stride = u.in_channels * weight_ic_stride;

    // Iterate over all output pixels that this input pixel could have contributed to.
    // Forward mapping: oy = (iy + pad - ky) / stride, ox = (ix + pad - kx) / stride
    // We iterate over the kernel [ky, kx] and see if the resulting [oy, ox] is valid.
    
    for (var oc: u32 = 0u; oc < u.out_channels; oc = oc + 1u) {
        let out_c_offset = oc * u.out_height * u.out_width;
        let weight_oc_offset = oc * weight_oc_stride + ic * weight_ic_stride;
        
        for (var ky: u32 = 0u; ky < u.kernel_h; ky = ky + 1u) {
            let py = iy + u.pad;
            if (py < ky) { continue; }
            let y_after_pad = py - ky;
            if (y_after_pad % u.stride != 0u) { continue; }
            let oy = y_after_pad / u.stride;
            if (oy >= u.out_height) { continue; }

            for (var kx: u32 = 0u; kx < u.kernel_w; kx = kx + 1u) {
                let px = ix + u.pad;
                if (px < kx) { continue; }
                let x_after_pad = px - kx;
                if (x_after_pad % u.stride != 0u) { continue; }
                let ox = x_after_pad / u.stride;
                if (ox >= u.out_width) { continue; }

                let dy = grad_output[out_c_offset + oy * u.out_width + ox];
                let w = weight[weight_oc_offset + ky * u.kernel_w + kx];
                sum = sum + dy * w;
            }
        }
    }

    grad_input[in_idx] = sum;
}
