// vision_average_pool.wgsl

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    grid_height: u32,
    grid_width: u32,
    hidden_size: u32,
    pooling_size: u32,
    pooled_height: u32,
    pooled_width: u32,
    output_elements: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_idx = gid.x;
    if (output_idx >= u.output_elements) {
        return;
    }

    let pooled_idx = output_idx / u.hidden_size;
    let hidden_idx = output_idx % u.hidden_size;
    let pooled_y = pooled_idx / u.pooled_width;
    let pooled_x = pooled_idx % u.pooled_width;
    var sum = 0.0f;
    for (var local_y = 0u; local_y < u.pooling_size; local_y = local_y + 1u) {
        for (var local_x = 0u; local_x < u.pooling_size; local_x = local_x + 1u) {
            let source_y = pooled_y * u.pooling_size + local_y;
            let source_x = pooled_x * u.pooling_size + local_x;
            let source_idx = (source_y * u.grid_width + source_x) * u.hidden_size + hidden_idx;
            sum = sum + input[source_idx];
        }
    }
    let divisor = f32(u.pooling_size * u.pooling_size);
    output[output_idx] = (sum / divisor) * sqrt(f32(u.hidden_size));
}
