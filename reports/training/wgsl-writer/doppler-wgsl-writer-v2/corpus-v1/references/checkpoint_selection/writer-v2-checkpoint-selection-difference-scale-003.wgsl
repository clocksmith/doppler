override WORKGROUP_SIZE: u32 = 32u;

struct Uniforms {
    logical_size: u32,
    write_offset: u32,
    scale: f32,
    second_reserved: f32,
}

@group(0) @binding(0) var<storage, read> a_values: array<f32>;
@group(0) @binding(1) var<storage, read> b_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> result_data: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) dispatch_id: vec3<u32>) {
    let idx = dispatch_id.x;
    if (idx >= uniforms.logical_size) {
        return;
    }

    let value = (a_values[idx] - b_values[idx]) * uniforms.scale;
    result_data[uniforms.write_offset + idx] = value;
}
