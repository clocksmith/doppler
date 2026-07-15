override WORKGROUP_SIZE: u32 = 256u;

struct KernelParams {
    num_elements: u32,
    destination_start: u32,
    scale: f32,
    second_reserved: f32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> destination: array<f32>;
@group(0) @binding(3) var<uniform> kernel_params: KernelParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let linear_index = invocation.x;
    if (linear_index >= kernel_params.num_elements) {
        return;
    }

    let computed = (left[linear_index] - right[linear_index]) * kernel_params.scale;
    destination[kernel_params.destination_start + linear_index] = computed;
}
