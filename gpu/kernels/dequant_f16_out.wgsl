// Q4_K Dequantization Kernel - f16 Output
//
// Dequantizes Q4_K blocks directly to f16 to avoid an intermediate f32 buffer
// when weights will be consumed by f16-weight matmul kernels.
//
// llama.cpp Q4_K format:
//   - 256 elements per super-block
//   - 8 sub-blocks of 32 elements each
//   - 12 bytes encode 8 scales + 8 mins (6 bits each, packed)
//   - 128 bytes of 4-bit quantized values

enable f16;

// Q4_K constants
const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;

struct Uniforms {
    num_blocks: u32,
    output_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Q4KBlock {
    d: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> quantized: array<Q4KBlock>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

var<workgroup> shared_scales: array<f32, 8>;
var<workgroup> shared_mins: array<f32, 8>;
var<workgroup> shared_d: f32;
var<workgroup> shared_dmin: f32;

fn unpack_f16_lo(packed: u32) -> f32 {
    return unpack2x16float(packed).x;
}

fn unpack_f16_hi(packed: u32) -> f32 {
    return unpack2x16float(packed).y;
}

// Get byte from scales array (12 bytes packed as 3 u32)
fn get_scale_byte(scales: array<u32, 3>, byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    return (scales[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

// llama.cpp Q4_K scale/min extraction (get_scale_min_k4):
// For sub-blocks 0-3: simple 6-bit extraction
// For sub-blocks 4-7: complex packing with upper bits from earlier bytes
fn get_scale_min_k4(scales: array<u32, 3>, j: u32) -> vec2<u32> {
    var sc: u32;
    var mn: u32;

    if (j < 4u) {
        // Simple case: lower 6 bits
        sc = get_scale_byte(scales, j) & 63u;
        mn = get_scale_byte(scales, j + 4u) & 63u;
    } else {
        // Complex case: 4 bits from bytes 8-11, upper 2 bits from bytes 0-7
        let q_j = get_scale_byte(scales, j + 4u);  // bytes 8-11
        let q_lo = get_scale_byte(scales, j - 4u); // bytes 0-3 (for upper bits of scale)
        let q_hi = get_scale_byte(scales, j);      // bytes 4-7 (for upper bits of min)

        sc = (q_j & 0xFu) | ((q_lo >> 6u) << 4u);
        mn = (q_j >> 4u) | ((q_hi >> 6u) << 4u);
    }

    return vec2<u32>(sc, mn);
}

// Extract 4-bit quantized value
// Q4_K nibble layout per 64-element chunk:
//   - Elements 0-31: lower nibbles of 32 bytes
//   - Elements 32-63: upper nibbles of same 32 bytes
// Layout: chunk0 (elem 0-63) uses bytes 0-31
//         chunk1 (elem 64-127) uses bytes 32-63
//         chunk2 (elem 128-191) uses bytes 64-95
//         chunk3 (elem 192-255) uses bytes 96-127
fn get_q4(qs: array<u32, 32>, idx: u32) -> u32 {
    // Which 64-element chunk? (0-3)
    let chunk = idx / 64u;
    // Position within chunk (0-63)
    let pos_in_chunk = idx % 64u;
    // First or second half of chunk?
    let use_upper = pos_in_chunk >= 32u;
    // Byte index within the 32-byte range for this chunk
    let byte_in_range = select(pos_in_chunk, pos_in_chunk - 32u, use_upper);
    // Absolute byte index
    let byte_idx = chunk * 32u + byte_in_range;

    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    let byte_val = (qs[word_idx] >> (byte_in_word * 8u)) & 0xFFu;

    if (use_upper) {
        return (byte_val >> 4u) & 0xFu;
    } else {
        return byte_val & 0xFu;
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let block_idx = workgroup_id.x;
    let elem_idx = local_id.x;

    if (block_idx >= uniforms.num_blocks) {
        return;
    }

    let block = quantized[block_idx];

    if (elem_idx == 0u) {
        shared_d = unpack_f16_lo(block.d);
        shared_dmin = unpack_f16_hi(block.d);
    }

    // Threads 0-7 load scales and mins for 8 sub-blocks
    if (elem_idx < 8u) {
        let sm = get_scale_min_k4(block.scales, elem_idx);
        shared_scales[elem_idx] = f32(sm.x);
        shared_mins[elem_idx] = f32(sm.y);
    }

    workgroupBarrier();

    let subblock_idx = elem_idx / SUBBLOCK_SIZE;  // 0-7
    let scale = shared_d * shared_scales[subblock_idx];
    let min_val = shared_dmin * shared_mins[subblock_idx];

    let q = get_q4(block.qs, elem_idx);
    // llama.cpp formula: dequant = d * scale * q - dmin * min
    let dequant = scale * f32(q) - min_val;

    let out_idx = uniforms.output_offset + block_idx * QK_K + elem_idx;
    output[out_idx] = f16(dequant);
}

@compute @workgroup_size(64, 1, 1)
fn main_vec4(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let block_idx = workgroup_id.x;
    let thread_idx = local_id.x;

    if (block_idx >= uniforms.num_blocks) {
        return;
    }

    let block = quantized[block_idx];

    if (thread_idx == 0u) {
        shared_d = unpack_f16_lo(block.d);
        shared_dmin = unpack_f16_hi(block.d);
    }

    // Threads 0-7 load scales and mins for 8 sub-blocks
    if (thread_idx < 8u) {
        let sm = get_scale_min_k4(block.scales, thread_idx);
        shared_scales[thread_idx] = f32(sm.x);
        shared_mins[thread_idx] = f32(sm.y);
    }

    workgroupBarrier();

    let d = shared_d;
    let dmin = shared_dmin;

    let base_elem = thread_idx * 4u;
    let subblock_idx = base_elem / SUBBLOCK_SIZE;  // 0-7
    let scale = d * shared_scales[subblock_idx];
    let min_val = dmin * shared_mins[subblock_idx];

    let out_base = uniforms.output_offset + block_idx * QK_K + base_elem;

    // llama.cpp formula: dequant = d * scale * q - dmin * min
    output[out_base + 0u] = f16(scale * f32(get_q4(block.qs, base_elem + 0u)) - min_val);
    output[out_base + 1u] = f16(scale * f32(get_q4(block.qs, base_elem + 1u)) - min_val);
    output[out_base + 2u] = f16(scale * f32(get_q4(block.qs, base_elem + 2u)) - min_val);
    output[out_base + 3u] = f16(scale * f32(get_q4(block.qs, base_elem + 3u)) - min_val);
}
