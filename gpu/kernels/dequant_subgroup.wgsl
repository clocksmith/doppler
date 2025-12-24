// Q4_K Dequantization Kernel - Subgroup Optimized
//
// Dequantizes Q4_K quantized weights using subgroup broadcast operations.
//
// llama.cpp Q4_K format:
//   - 256 elements per super-block
//   - 8 sub-blocks of 32 elements each
//   - 12 bytes encode 8 scales + 8 mins (6 bits each, packed)
//   - 128 bytes of 4-bit quantized values
//
// Subgroup operations enable efficient broadcast of scales to all lanes,
// reducing memory reads and improving throughput.

enable subgroups;

// Q4_K constants
const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;

// Uniforms
struct Uniforms {
    num_blocks: u32,    // Total number of Q4_K blocks
    output_offset: u32, // Offset in output buffer
    _pad0: u32,
    _pad1: u32,
}

// Q4_K block structure (packed layout matching llama.cpp)
// Total size per block: 144 bytes
//   - d (f16): super-block scale (2 bytes)
//   - dmin (f16): super-block min (2 bytes)
//   - scales (12 bytes): packed 6-bit scales for sub-blocks
//   - qs (128 bytes): 4-bit quantized values (256 * 4 bits / 8 = 128 bytes)
struct Q4KBlock {
    d: u32,           // d and dmin packed as f16 pair
    scales: array<u32, 3>, // 12 bytes of packed scales
    qs: array<u32, 32>,    // 128 bytes of quantized values
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> quantized: array<Q4KBlock>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Extract f16 from packed u32
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

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let block_idx = global_id.x / QK_K;
    let elem_idx = global_id.x % QK_K;

    // Use block 0 for out-of-bounds threads to maintain uniform control flow
    // (required for subgroup operations)
    let safe_block_idx = select(block_idx, 0u, block_idx >= uniforms.num_blocks);
    let in_bounds = block_idx < uniforms.num_blocks;

    let block = quantized[safe_block_idx];

    // Extract super-block scale and min
    let d = unpack_f16_lo(block.d);
    let dmin = unpack_f16_hi(block.d);

    // Determine sub-block (8 sub-blocks of 32 elements each)
    let subblock_idx = elem_idx / SUBBLOCK_SIZE;

    // Get sub-block scale (use subgroup broadcast for efficiency)
    // Leader thread in each subgroup reads the scale
    var scale: f32;
    var min_val: f32;

    // Subgroup broadcast: thread 0 reads, broadcasts to all
    let is_leader = (sg_id == 0u);
    if (is_leader) {
        let sm = get_scale_min_k4(block.scales, subblock_idx);
        scale = d * f32(sm.x);
        min_val = dmin * f32(sm.y);
    }

    // Broadcast scale and min to all threads in subgroup
    scale = subgroupBroadcastFirst(scale);
    min_val = subgroupBroadcastFirst(min_val);

    // Get quantized value
    let q = get_q4(block.qs, elem_idx);

    // llama.cpp formula: dequant = d * scale * q - dmin * min
    let dequant = scale * f32(q) - min_val;

    // Write output only for in-bounds threads
    if (in_bounds) {
        let out_idx = uniforms.output_offset + block_idx * QK_K + elem_idx;
        output[out_idx] = dequant;
    }
}

// Entry point for processing multiple elements per thread (4x unroll)
// Supports 2D dispatch for large tensors (>65535 workgroups)
@compute @workgroup_size(64, 1, 1)
fn main_vec4(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    // Support 2D dispatch: thread_idx = global_id.x + global_id.y * numWorkgroupsX * 64
    let thread_idx = global_id.x + global_id.y * num_wg.x * 64u;
    let block_idx = thread_idx / 64u;  // 64 threads per block (256/4 elements each)
    let local_idx = thread_idx % 64u;

    // Use block 0 for out-of-bounds threads to maintain uniform control flow
    // (required for subgroup operations)
    let safe_block_idx = select(block_idx, 0u, block_idx >= uniforms.num_blocks);
    let in_bounds = block_idx < uniforms.num_blocks;

    let block = quantized[safe_block_idx];
    let d = unpack_f16_lo(block.d);
    let dmin = unpack_f16_hi(block.d);

    // Each thread processes 4 consecutive elements
    let base_elem = local_idx * 4u;
    let subblock_idx = base_elem / SUBBLOCK_SIZE;  // 0-7

    // Broadcast scales within subgroup
    var scale: f32;
    var min_val: f32;

    let is_leader = (sg_id % 8u == 0u);  // Leader per sub-block worth of threads
    if (is_leader) {
        let sm = get_scale_min_k4(block.scales, subblock_idx);
        scale = d * f32(sm.x);
        min_val = dmin * f32(sm.y);
    }
    scale = subgroupBroadcastFirst(scale);
    min_val = subgroupBroadcastFirst(min_val);

    // Process 4 elements (only write if in bounds)
    // llama.cpp formula: dequant = d * scale * q - dmin * min
    if (in_bounds) {
        let out_base = uniforms.output_offset + block_idx * QK_K + base_elem;

        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let q = get_q4(block.qs, base_elem + i);
            output[out_base + i] = scale * f32(q) - min_val;
        }
    }
}
