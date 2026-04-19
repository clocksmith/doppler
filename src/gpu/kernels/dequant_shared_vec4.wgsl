// Q4_K Dequantization Kernel - Shared Memory Fallback (vec4)
//
// Vectorized variant for workgroup size 64.

// Q4_K constants
const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;

// Tunable workgroup size
override WORKGROUP_SIZE_VEC4: u32 = 64u;

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

// Vectorized version - each thread handles 4 elements
// Workgroup processes one block with 64 threads
@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn main_vec4(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    // Support 2D dispatch for tensors with >65535 blocks.
    let block_idx = workgroup_id.x + workgroup_id.y * num_wg.x;
    let thread_idx = local_id.x;

    if (block_idx >= u.num_blocks) {
        return;
    }

    let block = quantized[block_idx];
    let scale_word0 = block.scales[0];
    let scale_word1 = block.scales[1];
    let scale_word2 = block.scales[2];

    // Load shared data
    if (thread_idx == 0u) {
        shared_d = unpack_f16_lo(block.d);
        shared_dmin = unpack_f16_hi(block.d);
    }

    // Threads 0-7 load scales and mins for 8 sub-blocks
    if (thread_idx < 8u) {
        let sm = get_scale_min_k4(scale_word0, scale_word1, scale_word2, thread_idx);
        shared_scales[thread_idx] = f32(sm.x);
        shared_mins[thread_idx] = f32(sm.y);
    }

    workgroupBarrier();

    let d = shared_d;
    let dmin = shared_dmin;

    // Each thread processes 4 elements
    let base_elem = thread_idx * 4u;
    let subblock_idx = base_elem / SUBBLOCK_SIZE;  // 0-7
    let scale = d * shared_scales[subblock_idx];
    let min_val = dmin * shared_mins[subblock_idx];

    let out_base = u.output_offset + block_idx * QK_K + base_elem;

    // llama.cpp formula: dequant = d * scale * q - dmin * min
    let chunk0 = (base_elem + 0u) / 64u;
    let pos0 = (base_elem + 0u) % 64u;
    let upper0 = pos0 >= 32u;
    let byte0 = chunk0 * 32u + select(pos0, pos0 - 32u, upper0);
    let byte_val0 = (block.qs[byte0 / 4u] >> ((byte0 % 4u) * 8u)) & 0xFFu;
    let q0 = select(byte_val0 & 0xFu, (byte_val0 >> 4u) & 0xFu, upper0);

    let chunk1 = (base_elem + 1u) / 64u;
    let pos1 = (base_elem + 1u) % 64u;
    let upper1 = pos1 >= 32u;
    let byte1 = chunk1 * 32u + select(pos1, pos1 - 32u, upper1);
    let byte_val1 = (block.qs[byte1 / 4u] >> ((byte1 % 4u) * 8u)) & 0xFFu;
    let q1 = select(byte_val1 & 0xFu, (byte_val1 >> 4u) & 0xFu, upper1);

    let chunk2 = (base_elem + 2u) / 64u;
    let pos2 = (base_elem + 2u) % 64u;
    let upper2 = pos2 >= 32u;
    let byte2 = chunk2 * 32u + select(pos2, pos2 - 32u, upper2);
    let byte_val2 = (block.qs[byte2 / 4u] >> ((byte2 % 4u) * 8u)) & 0xFFu;
    let q2 = select(byte_val2 & 0xFu, (byte_val2 >> 4u) & 0xFu, upper2);

    let chunk3 = (base_elem + 3u) / 64u;
    let pos3 = (base_elem + 3u) % 64u;
    let upper3 = pos3 >= 32u;
    let byte3 = chunk3 * 32u + select(pos3, pos3 - 32u, upper3);
    let byte_val3 = (block.qs[byte3 / 4u] >> ((byte3 % 4u) * 8u)) & 0xFFu;
    let q3 = select(byte_val3 & 0xFu, (byte_val3 >> 4u) & 0xFu, upper3);

    output[out_base + 0u] = scale * f32(q0) - min_val;
    output[out_base + 1u] = scale * f32(q1) - min_val;
    output[out_base + 2u] = scale * f32(q2) - min_val;
    output[out_base + 3u] = scale * f32(q3) - min_val;
}
