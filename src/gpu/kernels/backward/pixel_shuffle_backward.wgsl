// pixel_shuffle_backward.wgsl
//
// Inverse of Pixel Shuffle: Rearranges [C, H, W] -> [H/P, W/P, C*P*P]
// 
// Forward: tokens [grid_height * grid_width, patch_channels] -> output [out_channels, out_height, out_width]
// Backward: grad_output [out_channels, out_height, out_width] -> grad_input [grid_height * grid_width, patch_channels]

struct Uniforms {
    out_channels: u32,
    out_height: u32,
    out_width: u32,
    grid_width: u32,
    grid_height: u32,
    patch_size: u32,
    patch_channels: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

override WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let input_idx = gid.x;
    let total_input_elements = u.grid_width * u.grid_height * u.patch_channels;
    
    if (input_idx >= total_input_elements) {
        return;
    }

    // Resolve input coordinates (inverse of forward mapping)
    let token_idx = input_idx / u.patch_channels;
    let patch_idx = input_idx % u.patch_channels;
    
    let grid_y = token_idx / u.grid_width;
    let grid_x = token_idx % u.grid_width;
    
    // patch_idx = (subY * patchSize + subX) * outChannels + c
    let c = patch_idx % u.out_channels;
    let sub_rem = patch_idx / u.out_channels;
    let sub_y = sub_rem / u.patch_size;
    let sub_x = sub_rem % u.patch_size;
    
    let y = grid_y * u.patch_size + sub_y;
    let x = grid_x * u.patch_size + sub_x;
    
    let spatial = u.out_height * u.out_width;
    let output_idx = (c * u.out_height + y) * u.out_width + x;
    
    grad_input[input_idx] = grad_output[output_idx];
}
