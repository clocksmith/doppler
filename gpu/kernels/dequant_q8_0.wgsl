// Q8_0 Dequantization Kernel - f16 Output
//
// Dequantizes Q8_0 blocks (8-bit quantization from llama.cpp/GGUF).
//
// Q8_0 block layout (34 bytes per 32 elements):
//   - d: 2 bytes at offset 0 (f16 scale)
//   - qs: 32 bytes at offset 2 (int8 quantized values)
//
// Algorithm from ggml-quants.c dequantize_row_q8_0

enable f16;

// Q8_0 constants
const QK8_0: u32 = 32u;
const Q8_0_BLOCK_BYTES: u32 = 34u;

// Byte offsets in Q8_0 block
const D_OFFSET: u32 = 0u;   // 2 bytes (f16 scale)
const QS_OFFSET: u32 = 2u;  // 32 bytes (int8 values)

// Tunable workgroup size
override WORKGROUP_SIZE: u32 = 32u;

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

// Read a byte from the quantized buffer at a given block and byte offset
fn read_byte(block_idx: u32, byte_offset: u32) -> u32 {
    let global_byte = block_idx * Q8_0_BLOCK_BYTES + byte_offset;
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

    // Thread 0 loads d (f16 at offset 0)
    if (elem_idx == 0u) {
        let d_packed = read_u16(block_idx, D_OFFSET);
        shared_d = unpack_f16(d_packed);
    }

    workgroupBarrier();

    let d = shared_d;

    // Read quantized value as signed int8
    let q = read_i8_as_f32(block_idx, QS_OFFSET + elem_idx);

    // Dequantize: output = d * q
    let dequant = d * q;

    let out_idx = u.output_offset + block_idx * QK8_0 + elem_idx;
    output[out_idx] = f16(dequant);
}
