override WORKGROUP_SIZE: u32 = 32u;

struct Config {
    element_count: u32,
    destination_offset: u32,
    first_reserved: f32,
    second_reserved: f32,
}

@group(0) @binding(0) var<storage, read> lhs_values: array<f32>;
@group(0) @binding(1) var<storage, read> rhs_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> destination_values: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let element_index = invocation_id.x;
    if (element_index >= config.element_count) {
        return;
    }

    let computed_value = abs(lhs_values[element_index] - rhs_values[element_index]);
    destination_values[config.destination_offset + element_index] = computed_value;
}
