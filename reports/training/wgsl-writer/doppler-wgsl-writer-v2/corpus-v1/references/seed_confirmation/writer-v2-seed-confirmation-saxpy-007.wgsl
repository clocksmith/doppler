override WORKGROUP_SIZE: u32 = 128u;

struct DispatchParams {
    active_length: u32,
    output_base: u32,
    scale: f32,
    second_reserved: f32,
}

@group(0) @binding(0) var<storage, read> x_values: array<f32>;
@group(0) @binding(1) var<storage, read> y_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(3) var<uniform> dispatch_params: DispatchParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_position: vec3<u32>) {
    let position = global_position.x;
    if (position >= dispatch_params.active_length) {
        return;
    }

    let answer = x_values[position] + dispatch_params.scale * y_values[position];
    output_buffer[dispatch_params.output_base + position] = answer;
}
