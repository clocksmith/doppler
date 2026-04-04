// Depthwise Conv1D Kernel
//
// Applies depthwise 1D convolution along the time axis.
// Input:  [channels, length]
// Weight: [channels, 1, kernelSize]
// Output: [channels, length] (same-padded)

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    channels: u32,
    length: u32,
    kernel_size: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spatial_idx = gid.x;
    let ch = gid.y;
    if (ch >= u.channels || spatial_idx >= u.length) {
        return;
    }

    let half_k = i32(u.kernel_size) / 2;
    let length = i32(u.length);
    let t = i32(spatial_idx);

    var sum: f32 = 0.0;
    let weight_offset = ch * u.kernel_size;
    let input_offset = ch * u.length;

    for (var k: u32 = 0u; k < u.kernel_size; k = k + 1u) {
        let in_t = t + i32(k) - half_k;
        if (in_t >= 0 && in_t < length) {
            sum = sum + input[input_offset + u32(in_t)] * weight[weight_offset + k];
        }
    }

    output[ch * u.length + spatial_idx] = sum;
}
