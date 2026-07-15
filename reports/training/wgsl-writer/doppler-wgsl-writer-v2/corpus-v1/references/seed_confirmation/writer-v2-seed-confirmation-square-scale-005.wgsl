override WORKGROUP_SIZE: u32 = 256u;

struct Parameters {
    value_count: u32,
    output_start: u32,
    scale: f32,
    second_reserved: f32,
}

@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let item_index = global_id.x;
    if (item_index >= parameters.value_count) {
        return;
    }

    let transformed = (samples[item_index] * samples[item_index]) * parameters.scale;
    outputs[parameters.output_start + item_index] = transformed;
}
