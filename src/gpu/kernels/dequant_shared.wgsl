// Q4_K Dequantization Kernel - Shared Memory Fallback
//
// Dequantizes Q4_K quantized weights using workgroup shared memory.
// This is the fallback when subgroup operations are unavailable.
//
// llama.cpp Q4_K format:
//   - 256 elements per super-block
//   - 8 sub-blocks of 32 elements each
//   - 12 bytes encode 8 scales + 8 mins (6 bits each, packed)
//   - 128 bytes of 4-bit quantized values

// Q4_K constants
const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;

// Tunable workgroup sizes
override WORKGROUP_SIZE_MAIN: u32 = 256u;
override WORKGROUP_SIZE_F16: u32 = 256u;

struct Uniforms {
    num_blocks: u32,
    output_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Q4KBlock {
    d: u32,                    // d (f16) and dmin (f16) packed
    scales: array<u32, 3>,     // 12 bytes of packed scales/mins
    qs: array<u32, 32>,        // 128 bytes of 4-bit values
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> quantized: array<Q4KBlock>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Shared memory for 8 sub-blocks
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

// Get byte from the 12-byte scales payload packed as 3 u32 words.
fn get_scale_byte(scale_word0: u32, scale_word1: u32, scale_word2: u32, byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    let word = select(
        select(scale_word0, scale_word1, word_idx == 1u),
        scale_word2,
        word_idx == 2u
    );
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

// llama.cpp Q4_K scale/min extraction (get_scale_min_k4):
// For sub-blocks 0-3: simple 6-bit extraction
// For sub-blocks 4-7: complex packing with upper bits from earlier bytes
fn get_scale_min_k4(scale_word0: u32, scale_word1: u32, scale_word2: u32, j: u32) -> vec2<u32> {
    var sc: u32;
    var mn: u32;

    if (j < 4u) {
        // Simple case: lower 6 bits
        sc = get_scale_byte(scale_word0, scale_word1, scale_word2, j) & 63u;
        mn = get_scale_byte(scale_word0, scale_word1, scale_word2, j + 4u) & 63u;
    } else {
        // Complex case: 4 bits from bytes 8-11, upper 2 bits from bytes 0-7
        let q_j = get_scale_byte(scale_word0, scale_word1, scale_word2, j + 4u);  // bytes 8-11
        let q_lo = get_scale_byte(scale_word0, scale_word1, scale_word2, j - 4u); // bytes 0-3 (for upper bits of scale)
        let q_hi = get_scale_byte(scale_word0, scale_word1, scale_word2, j);      // bytes 4-7 (for upper bits of min)

        sc = (q_j & 0xFu) | ((q_lo >> 6u) << 4u);
        mn = (q_j >> 4u) | ((q_hi >> 6u) << 4u);
    }

    return vec2<u32>(sc, mn);
}

@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    // Support 2D dispatch for tensors with >65535 blocks.
    let block_idx = workgroup_id.x + workgroup_id.y * num_wg.x;
    let elem_idx = local_id.x;

    if (block_idx >= u.num_blocks) {
        return;
    }

    let block = quantized[block_idx];
    let scale_word0 = block.scales[0];
    let scale_word1 = block.scales[1];
    let scale_word2 = block.scales[2];

    // First thread loads d and dmin
    if (elem_idx == 0u) {
        shared_d = unpack_f16_lo(block.d);
        shared_dmin = unpack_f16_hi(block.d);
    }

    // Threads 0-7 load scales and mins for all 8 sub-blocks
    if (elem_idx < 8u) {
        let sm = get_scale_min_k4(scale_word0, scale_word1, scale_word2, elem_idx);
        shared_scales[elem_idx] = f32(sm.x);
        shared_mins[elem_idx] = f32(sm.y);
    }

    // Wait for shared memory to be populated
    workgroupBarrier();

    // Now all threads can read scales efficiently
    let d = shared_d;
    let dmin = shared_dmin;
    let subblock_idx = elem_idx / SUBBLOCK_SIZE;  // 0-7 for 8 sub-blocks of 32
    let scale = d * shared_scales[subblock_idx];
    let min_val = dmin * shared_mins[subblock_idx];

    // Get quantized value and dequantize
    // llama.cpp formula: dequant = d * scale * q - dmin * min
    let chunk = elem_idx / 64u;
    let pos_in_chunk = elem_idx % 64u;
    let use_upper = pos_in_chunk >= 32u;
    let byte_in_range = select(pos_in_chunk, pos_in_chunk - 32u, use_upper);
    let byte_idx = chunk * 32u + byte_in_range;
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    let byte_val = (block.qs[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
    let q = select(byte_val & 0xFu, (byte_val >> 4u) & 0xFu, use_upper);
    let dequant = scale * f32(q) - min_val;

    // Write output
    let out_idx = u.output_offset + block_idx * QK_K + elem_idx;
    output[out_idx] = dequant;
}

// FP16 output variant for when downstream consumers want f16
@compute @workgroup_size(WORKGROUP_SIZE_F16, 1, 1)
fn main_f16_out(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let block_idx = workgroup_id.x;
    let elem_idx = local_id.x;

    if (block_idx >= u.num_blocks) {
        return;
    }

    let block = quantized[block_idx];
    let scale_word0 = block.scales[0];
    let scale_word1 = block.scales[1];
    let scale_word2 = block.scales[2];

    if (elem_idx == 0u) {
        shared_d = unpack_f16_lo(block.d);
        shared_dmin = unpack_f16_hi(block.d);
    }

    // Threads 0-7 load scales and mins
    if (elem_idx < 8u) {
        let sm = get_scale_min_k4(scale_word0, scale_word1, scale_word2, elem_idx);
        shared_scales[elem_idx] = f32(sm.x);
        shared_mins[elem_idx] = f32(sm.y);
    }

    workgroupBarrier();

    let subblock_idx = elem_idx / SUBBLOCK_SIZE;  // 0-7
    let scale = shared_d * shared_scales[subblock_idx];
    let min_val = shared_dmin * shared_mins[subblock_idx];

    let chunk = elem_idx / 64u;
    let pos_in_chunk = elem_idx % 64u;
    let use_upper = pos_in_chunk >= 32u;
    let byte_in_range = select(pos_in_chunk, pos_in_chunk - 32u, use_upper);
    let byte_idx = chunk * 32u + byte_in_range;
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    let byte_val = (block.qs[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
    let q = select(byte_val & 0xFu, (byte_val >> 4u) & 0xFu, use_upper);
    // llama.cpp formula: dequant = d * scale * q - dmin * min
    let dequant = scale * f32(q) - min_val;

    // Note: This writes f32, but caller can cast if needed
    let out_idx = u.output_offset + block_idx * QK_K + elem_idx;
    output[out_idx] = dequant;
}
