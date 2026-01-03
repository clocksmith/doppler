// upsample2d_backward.wgsl
//
// Computes gradient for Upsample2D (nearest).
// Since multiple output pixels map to one input pixel, 
// the input gradient is the sum of the gradients of all corresponding output pixels.

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
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

override WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_idx = gid.x;
    let total_in_elements = u.channels * u.in_height * u.in_width;
    
    if (in_idx >= total_in_elements) {
        return;
    }

    // Resolve input coordinates
    let in_spatial = u.in_height * u.in_width;
    let c = in_idx / in_spatial;
    let rem = in_idx % in_spatial;
    let in_y = rem / u.in_width;
    let in_x = rem % u.in_width;

    // Sum gradients from the scale * scale patch in the output
    var sum: f32 = 0.0;
    let out_y_start = in_y * u.scale;
    let out_x_start = in_x * u.scale;
    
    let out_spatial = u.out_height * u.out_width;
    let out_c_offset = c * out_spatial;

    for (var dy: u32 = 0u; dy < u.scale; dy = dy + 1u) {
        for (var dx: u32 = 0u; dx < u.scale; dx = dx + 1u) {
            let oy = out_y_start + dy;
            let ox = out_x_start + dx;
            if (oy < u.out_height && ox < u.out_width) {
                sum = sum + grad_output[out_c_offset + oy * u.out_width + ox];
            }
        }
    }

    grad_input[in_idx] = sum;
}
