// Q6_K Dequantization Kernel - f16 Output
//
// Dequantizes Q6_K blocks (6-bit quantization from llama.cpp/GGUF).
//
// Q6_K block layout (210 bytes per 256 elements):
//   - ql: 128 bytes at offset 0 (low 4 bits of quants)
//   - qh: 64 bytes at offset 128 (high 2 bits of quants)
//   - scales: 16 bytes at offset 192 (8-bit signed block scales)
//   - d: 2 bytes at offset 208 (f16 super-block scale)
//
// Algorithm from ggml-quants.c dequantize_row_q6_K

enable f16;

// Q6_K constants
const QK_K: u32 = 256u;
const Q6K_BLOCK_BYTES: u32 = 210u;

// Byte offsets in Q6_K block
const QL_OFFSET: u32 = 0u;      // 128 bytes
const QH_OFFSET: u32 = 128u;    // 64 bytes
const SCALES_OFFSET: u32 = 192u; // 16 bytes
const D_OFFSET: u32 = 208u;     // 2 bytes

// Tunable workgroup size
override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_blocks: u32,
    output_offset: u32,
    workgroups_x: u32,  // For 2D dispatch: blocks per row
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> quantized: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

var<workgroup> shared_d: f32;
var<workgroup> shared_scales: array<f32, 16>;

// Read a byte from the quantized buffer at a given block and byte offset
fn read_byte(block_idx: u32, byte_offset: u32) -> u32 {
    let global_byte = block_idx * Q6K_BLOCK_BYTES + byte_offset;
    let word_idx = global_byte / 4u;
    let byte_in_word = global_byte % 4u;
    return (quantized[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

// Read a u16 (little-endian) from the quantized buffer
fn read_u16(block_idx: u32, byte_offset: u32) -> u32 {
    let lo = read_byte(block_idx, byte_offset);
    let hi = read_byte(block_idx, byte_offset + 1u);
    return lo | (hi << 8u);
}

// Read signed i8 as f32
fn read_i8_as_f32(block_idx: u32, byte_offset: u32) -> f32 {
    let byte_val = read_byte(block_idx, byte_offset);
    // Convert u8 to i8 via two's complement
    if (byte_val >= 128u) {
        return f32(i32(byte_val) - 256);
    }
    return f32(byte_val);
}

// Unpack f16 from u32 (low 16 bits)
fn unpack_f16(packed: u32) -> f32 {
    return unpack2x16float(packed).x;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // Handle 2D dispatch for large block counts (> 65535)
    let block_idx = workgroup_id.x + workgroup_id.y * u.workgroups_x;
    let elem_idx = local_id.x;

    if (block_idx >= u.num_blocks) {
        return;
    }

    // Thread 0 loads d (f16 at offset 208)
    if (elem_idx == 0u) {
        let d_packed = read_u16(block_idx, D_OFFSET);
        shared_d = unpack_f16(d_packed);
    }

    // Threads 0-15 load scales (16 x i8 at offset 192)
    // Scales are signed i8 values used directly (no division)
    if (elem_idx < 16u) {
        shared_scales[elem_idx] = read_i8_as_f32(block_idx, SCALES_OFFSET + elem_idx);
    }

    workgroupBarrier();

    let d = shared_d;

    // Decompose element index into position within the block
    // Q6_K processes 256 elements in two 128-element halves
    // Each half has 4 quadrants of 32 elements
    let half = elem_idx / 128u;           // 0 or 1
    let within_half = elem_idx % 128u;    // 0..127
    let quadrant = within_half / 32u;     // 0, 1, 2, or 3
    let l = within_half % 32u;            // 0..31

    // ql index: l + 32*(quadrant&1) + 64*half
    // quadrant 0,2 use ql[l], quadrant 1,3 use ql[l+32]
    let ql_idx = l + 32u * (quadrant & 1u) + 64u * half;
    let ql_byte = read_byte(block_idx, QL_OFFSET + ql_idx);

    // ql shift: 0 for quadrant 0,1 (lower nibble); 4 for quadrant 2,3 (upper nibble)
    let ql_shift = 4u * (quadrant >> 1u);
    let ql_val = (ql_byte >> ql_shift) & 0xFu;

    // qh index: l + 32*half
    let qh_idx = l + 32u * half;
    let qh_byte = read_byte(block_idx, QH_OFFSET + qh_idx);

    // qh shift: 0 for q0, 2 for q1, 4 for q2, 6 for q3
    let qh_shift = quadrant * 2u;
    let qh_val = (qh_byte >> qh_shift) & 0x3u;

    // Combine to get 6-bit quantized value, then subtract 32
    let q6 = ql_val | (qh_val << 4u);
    let q = i32(q6) - 32;

    // Scale index from ggml.c: is + 2*quadrant + 8*half, where is = l/16
    let is_val = l / 16u;  // 0 or 1
    let scale_idx = is_val + 2u * quadrant + 8u * half;
    let scale = shared_scales[scale_idx];

    // Dequantize: output = d * scale * q
    let dequant = d * scale * f32(q);

    let out_idx = u.output_offset + block_idx * QK_K + elem_idx;
    output[out_idx] = f16(dequant);
}
