// Q4_K Dequantization Kernel - f32 Output, Row-Wise Matrix Layout
//
// Dequantizes Q4_K blocks to f32 with proper row-major matrix stride.
// Required when K (columns) is not aligned to QK_K (256), as the block
// layout has padding that must be skipped.
//
// Example: For K=1152, each row has 5 blocks (5*256=1280 elements stored),
// but only 1152 elements are valid. This kernel outputs [rows, K] shape
// with stride K, not stride blocksPerRow*256.

// Q4_K constants
const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;

// Tunable workgroup size
override WORKGROUP_SIZE_MAIN: u32 = 256u;

struct Uniforms {
    num_blocks: u32,
    blocks_per_row: u32,
    K: u32,               // actual columns (may not be 256-aligned)
    rows: u32,
}

struct Q4KBlock {
    d: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> quantized: array<Q4KBlock>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

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

fn get_scale_byte(scales: array<u32, 3>, byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    return (scales[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn get_scale_min_k4(scales: array<u32, 3>, j: u32) -> vec2<u32> {
    var sc: u32;
    var mn: u32;

    if (j < 4u) {
        sc = get_scale_byte(scales, j) & 63u;
        mn = get_scale_byte(scales, j + 4u) & 63u;
    } else {
        let q_j = get_scale_byte(scales, j + 4u);
        let q_lo = get_scale_byte(scales, j - 4u);
        let q_hi = get_scale_byte(scales, j);
        sc = (q_j & 0xFu) | ((q_lo >> 6u) << 4u);
        mn = (q_j >> 4u) | ((q_hi >> 6u) << 4u);
    }

    return vec2<u32>(sc, mn);
}

fn get_q4(qs: array<u32, 32>, idx: u32) -> u32 {
    let chunk = idx / 64u;
    let pos_in_chunk = idx % 64u;
    let use_upper = pos_in_chunk >= 32u;
    let byte_in_range = select(pos_in_chunk, pos_in_chunk - 32u, use_upper);
    let byte_idx = chunk * 32u + byte_in_range;

    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    let byte_val = (qs[word_idx] >> (byte_in_word * 8u)) & 0xFFu;

    if (use_upper) {
        return (byte_val >> 4u) & 0xFu;
    }
    return byte_val & 0xFu;
}

@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let block_idx = workgroup_id.x;
    let elem_idx = local_id.x;

    if (block_idx >= u.num_blocks) {
        return;
    }

    let block = quantized[block_idx];

    if (elem_idx == 0u) {
        shared_d = unpack_f16_lo(block.d);
        shared_dmin = unpack_f16_hi(block.d);
    }

    if (elem_idx < 8u) {
        let sm = get_scale_min_k4(block.scales, elem_idx);
        shared_scales[elem_idx] = f32(sm.x);
        shared_mins[elem_idx] = f32(sm.y);
    }

    workgroupBarrier();

    let row = block_idx / u.blocks_per_row;
    let block_in_row = block_idx % u.blocks_per_row;
    let col = block_in_row * QK_K + elem_idx;

    if (col >= u.K) {
        return;
    }

    let subblock_idx = elem_idx / SUBBLOCK_SIZE;
    let scale = shared_d * shared_scales[subblock_idx];
    let min_val = shared_dmin * shared_mins[subblock_idx];

    let q = get_q4(block.qs, elem_idx);
    let dequant = scale * f32(q) - min_val;

    let out_idx = row * u.K + col;
    output[out_idx] = dequant;
}
