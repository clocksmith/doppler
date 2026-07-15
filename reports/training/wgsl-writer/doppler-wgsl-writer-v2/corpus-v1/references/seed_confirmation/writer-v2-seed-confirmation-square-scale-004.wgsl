override WORKGROUP_SIZE: u32 = 128u;

struct Settings {
    item_count: u32,
    result_offset: u32,
    scale: f32,
    second_reserved: f32,
}

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;
@group(0) @binding(2) var<uniform> settings: Settings;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation: vec3<u32>) {
    let logical_index = global_invocation.x;
    if (logical_index >= settings.item_count) {
        return;
    }

    let output_value = (values[logical_index] * values[logical_index]) * settings.scale;
    results[settings.result_offset + logical_index] = output_value;
}
