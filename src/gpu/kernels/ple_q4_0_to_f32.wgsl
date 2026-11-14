// GGML Q4_0: each 32-value block is an fp16 scale and 16 packed bytes.
// Blocks and 160-value rows may start halfway through a storage u32.
override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUPS_X: u32 = 65535u;

struct Uniforms {
    num_elements: u32,
    input_offset: u32,
    output_offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn byte_at(offset: u32) -> u32 {
    return (input[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x + global_id.y * MAX_WORKGROUPS_X * WORKGROUP_SIZE;
    if (i >= u.num_elements) { return; }
    let element = u.input_offset + i;
    let base = (element / 32u) * 18u;
    let lane = element % 32u;
    let scale = unpack2x16float(byte_at(base) | (byte_at(base + 1u) << 8u)).x;
    let packed = byte_at(base + 2u + lane % 16u);
    let quant = (packed >> ((lane / 16u) * 4u)) & 15u;
    output[u.output_offset + i] = scale * f32(i32(quant) - 8);
}
