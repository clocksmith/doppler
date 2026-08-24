// vision_patch_embed_f16w.wgsl

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    grid_height: u32,
    grid_width: u32,
    channels: u32,
    patch_size: u32,
    temporal_patch_size: u32,
    hidden_size: u32,
    has_bias: u32,
    output_elements: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f16>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_idx = gid.x;
    if (output_idx >= u.output_elements) {
        return;
    }

    let patch_idx = output_idx / u.hidden_size;
    let hidden_idx = output_idx % u.hidden_size;
    let patch_y = patch_idx / u.grid_width;
    let patch_x = patch_idx % u.grid_width;
    let image_height = u.grid_height * u.patch_size;
    let image_width = u.grid_width * u.patch_size;
    let spatial_patch_area = u.channels * u.patch_size * u.patch_size;
    let temporal_patch_area = u.temporal_patch_size * spatial_patch_area;

    var value = 0.0f;
    if (u.has_bias != 0u) {
        value = f32(bias[hidden_idx]);
    }

    for (var channel = 0u; channel < u.channels; channel = channel + 1u) {
        for (var local_y = 0u; local_y < u.patch_size; local_y = local_y + 1u) {
            for (var local_x = 0u; local_x < u.patch_size; local_x = local_x + 1u) {
                let image_y = patch_y * u.patch_size + local_y;
                let image_x = patch_x * u.patch_size + local_x;
                let image_idx = channel * image_height * image_width + image_y * image_width + image_x;
                let spatial_idx = channel * u.patch_size * u.patch_size + local_y * u.patch_size + local_x;
                for (var temporal = 0u; temporal < u.temporal_patch_size; temporal = temporal + 1u) {
                    let weight_idx = hidden_idx * temporal_patch_area + temporal * spatial_patch_area + spatial_idx;
                    value = value + image[image_idx] * f32(weights[weight_idx]);
                }
            }
        }
    }

    output[output_idx] = value;
}
