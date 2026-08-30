// vision_spatial_merge.wgsl

override WORKGROUP_SIZE: u32 = 256u;
override CHANNEL_FIRST: bool = false;
override INPUT_BLOCK_MAJOR: bool = false;

struct Uniforms {
    grid_height: u32,
    grid_width: u32,
    hidden_size: u32,
    merge_size: u32,
    merged_height: u32,
    merged_width: u32,
    output_elements: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let output_idx = gid.x + gid.y * num_wg.x * WORKGROUP_SIZE;
    if (output_idx >= u.output_elements) {
        return;
    }

    let concat_dim = u.merge_size * u.merge_size * u.hidden_size;
    let merged_idx = output_idx / concat_dim;
    let concat_idx = output_idx % concat_dim;
    let hidden_idx = select(concat_idx % u.hidden_size, concat_idx / (u.merge_size * u.merge_size), CHANNEL_FIRST);
    let patch_in_merge = select(concat_idx / u.hidden_size, concat_idx % (u.merge_size * u.merge_size), CHANNEL_FIRST);
    let local_y = patch_in_merge / u.merge_size;
    let local_x = patch_in_merge % u.merge_size;
    let merged_y = merged_idx / u.merged_width;
    let merged_x = merged_idx % u.merged_width;
    let source_y = merged_y * u.merge_size + local_y;
    let source_x = merged_x * u.merge_size + local_x;
    let raster_source_idx = (source_y * u.grid_width + source_x) * u.hidden_size + hidden_idx;
    let block_source_idx = (merged_idx * u.merge_size * u.merge_size + patch_in_merge) * u.hidden_size + hidden_idx;
    let source_idx = select(raster_source_idx, block_source_idx, INPUT_BLOCK_MAJOR);

    output[output_idx] = input[source_idx];
}
