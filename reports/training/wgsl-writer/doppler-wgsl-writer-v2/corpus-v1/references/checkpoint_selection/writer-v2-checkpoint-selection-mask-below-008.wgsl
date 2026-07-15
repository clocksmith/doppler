override WORKGROUP_SIZE: u32 = 32u;

struct Control {
    logical_length: u32,
    store_offset: u32,
    threshold: f32,
    second_reserved: f32,
}

@group(0) @binding(0) var<storage, read> operands: array<f32>;
@group(0) @binding(1) var<storage, read_write> result_values: array<f32>;
@group(0) @binding(2) var<uniform> control: Control;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let offset = invocation_id.x;
    if (offset >= control.logical_length) {
        return;
    }

    let operation_result = select(0.0, operands[offset], operands[offset] >= control.threshold);
    result_values[control.store_offset + offset] = operation_result;
}
