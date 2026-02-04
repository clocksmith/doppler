// Pixel shuffle (tokens -> CHW) f16

enable f16;

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
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let spatial_size = u.out_height * u.out_width;
    let total = u.out_channels * spatial_size;
    if (idx >= total) {
        return;
    }

    let c = idx / spatial_size;
    let spatial = idx - c * spatial_size;
    let y = spatial / u.out_width;
    let x = spatial - y * u.out_width;

    let grid_y = y / u.patch_size;
    let grid_x = x / u.patch_size;
    let sub_y = y - grid_y * u.patch_size;
    let sub_x = x - grid_x * u.patch_size;

    let token_idx = grid_y * u.grid_width + grid_x;
    let patch_idx = (sub_y * u.patch_size + sub_x) * u.out_channels + c;
    let input_idx = token_idx * u.patch_channels + patch_idx;

    output[idx] = input[input_idx];
}
