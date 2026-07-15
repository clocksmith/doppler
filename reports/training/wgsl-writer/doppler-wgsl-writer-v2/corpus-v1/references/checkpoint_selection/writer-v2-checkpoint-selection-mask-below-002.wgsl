override WORKGROUP_SIZE: u32 = 128u;

struct Config {
    element_count: u32,
    destination_offset: u32,
    threshold: f32,
    second_reserved: f32,
}

@group(0) @binding(0) var<storage, read> source_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> destination_values: array<f32>;
@group(0) @binding(2) var<uniform> config: Config;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let element_index = invocation_id.x;
    if (element_index >= config.element_count) {
        return;
    }

    let computed_value = select(0.0, source_values[element_index], source_values[element_index] >= config.threshold);
    destination_values[config.destination_offset + element_index] = computed_value;
}
