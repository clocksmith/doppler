// AUTO-GENERATED from src/gpu/kernels/codegen/q4_0_common.wgslinc.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms { num_tokens: u32, hidden_size: u32, vocab_size: u32, index_offset: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
override BLOCK_VALUES: u32 = 32u;
override BLOCK_BYTES: u32 = 18u;
override QUANT_FORMAT: u32 = 0u;
fn q40_byte(offset: u32) -> u32 {
    return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}
fn packed_half(offset: u32) -> f32 {
    return unpack2x16float(q40_byte(offset) | (q40_byte(offset + 1u) << 8u)).x;
}
fn packed_signed_byte(offset: u32) -> i32 {
    return bitcast<i32>(q40_byte(offset) << 24u) >> 24;
}
fn q40_value(row: u32, k: u32, blocks: u32) -> f32 {
    let offset = (row * blocks + k / BLOCK_VALUES) * BLOCK_BYTES;
    let local = k % BLOCK_VALUES;
    if (QUANT_FORMAT == 3u) {
        let section = (local % 128u) / 32u;
        let lane = local % 32u;
        let low = q40_byte(offset + (local / 128u) * 64u + (section % 2u) * 32u + lane);
        let high = q40_byte(offset + 128u + (local / 128u) * 32u + lane);
        let value = ((low >> ((section / 2u) * 4u)) & 15u) | (((high >> (section * 2u)) & 3u) << 4u);
        return packed_half(offset + 208u) * f32(packed_signed_byte(offset + 192u + local / 16u)) * f32(i32(value) - 32);
    }
    let scale = packed_half(offset);
    if (QUANT_FORMAT == 2u) { return scale * f32(packed_signed_byte(offset + 2u + local)); }
    if (QUANT_FORMAT == 4u) {
        // Q5_0: f16 scale, 32 high bits (one per value), 16 bytes of low nibbles.
        let high_bits = q40_byte(offset + 2u) | (q40_byte(offset + 3u) << 8u)
            | (q40_byte(offset + 4u) << 16u) | (q40_byte(offset + 5u) << 24u);
        let low = (q40_byte(offset + 6u + local % 16u) >> select(0u, 4u, local >= 16u)) & 15u;
        return scale * f32(i32(low | (((high_bits >> local) & 1u) << 4u)) - 16);
    }
    let data_offset = select(2u,4u,QUANT_FORMAT == 1u);
    let packed = q40_byte(offset + data_offset + local % 16u);
    let shift = select(0u,4u,local >= 16u);
    let value = (packed >> shift) & 15u;
    if (QUANT_FORMAT == 1u) { return scale * f32(value) + packed_half(offset + 2u); }
    return scale * f32(i32(value) - 8);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = (group.x + group.y * grid.x) * WORKGROUP_SIZE + lane;
    if (index >= u.num_tokens * u.hidden_size) { return; }
    let row = indices[u.index_offset + index / u.hidden_size];
    var value = 0.0f;
    if (row < u.vocab_size) { value = q40_value(row,index % u.hidden_size,u.hidden_size / BLOCK_VALUES); }
    output[index] = value;
}
