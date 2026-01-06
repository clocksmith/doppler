// Fused Q4_K Matmul Kernel - W4A16 (F16 output, batched)
//
// Batched version for prefill (M > 1) with f16 output.

enable f16;
enable subgroups;

// Q4_K constants
const QK_K: u32 = 256u;           // Elements per super-block
const BLOCK_SIZE: u32 = 144u;     // Bytes per Q4_K block
const SUBBLOCK_SIZE: u32 = 32u;   // Elements per sub-block

struct Uniforms {
    M: u32,                   // Always 1 for GEMV
    N: u32,                   // Output dimension
    K: u32,                   // Inner dimension (must be multiple of 256 for Q4_K)
    alpha: f32,
    num_blocks_per_row: u32,  // K / 256
    _pad0: u32,               // 16-byte alignment padding
    _pad1: u32,
    _pad2: u32,
}

// Q4_K block structure (144 bytes)
// Layout: d(2) + dmin(2) + scales(12) + qs(128)
struct Q4KBlock {
    d_dmin: u32,          // d (f16) and dmin (f16) packed
    scales: array<u32, 3>, // 12 bytes of packed 6-bit scales
    qs: array<u32, 32>,   // 128 bytes of 4-bit quantized values
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B_q4k: array<Q4KBlock>;
@group(0) @binding(4) var<storage, read_write> C_f16: array<f16>;

// Extract f16 from packed u32
fn unpack_f16_lo(packed: u32) -> f32 {
    return unpack2x16float(packed).x;
}

fn unpack_f16_hi(packed: u32) -> f32 {
    return unpack2x16float(packed).y;
}

// Get byte from scales array
fn get_scale_byte(scales: array<u32, 3>, byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    return (scales[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

// llama.cpp Q4_K scale/min extraction
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

// Extract 4-bit quantized value from qs array
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
    } else {
        return byte_val & 0xFu;
    }
}

// ============================================================================
// Batched version for prefill (M > 1)
// Uses 2D dispatch: workgroup (x,y) computes output C[y*TILE_M : (y+1)*TILE_M, x]
// Each workgroup computes TILE_M rows × 1 column
// ============================================================================

override TILE_M: u32 = 4u;
override THREADS_PER_COL: u32 = 64u;
const MAX_SUBGROUPS_PER_ROW: u32 = 64u;  // Support sg_size >= 1 (64/1 = 64)

// Shared memory for per-row subgroup reduction: 4 rows × 16 max subgroups = 64
var<workgroup> batched_wg_sums: array<f32, 256>;

@compute @workgroup_size(64, 4, 1)
fn main_batched_f16(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let local_id = lid.x;
    let row = wg_id.y * TILE_M + lid.y;
    let col = wg_id.x;
    let is_valid = row < u.M && col < u.N;

    var partial_sum: f32 = 0.0;

    if (is_valid) {
        let num_blocks = u.num_blocks_per_row;
        for (var b: u32 = local_id; b < num_blocks; b = b + THREADS_PER_COL) {
            let block = B_q4k[col * num_blocks + b];
            let d = unpack_f16_lo(block.d_dmin);
            let dmin = unpack_f16_hi(block.d_dmin);
            let k_base = b * QK_K;

            for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                let sm = get_scale_min_k4(block.scales, sb);
                let scale = d * f32(sm.x);
                let min_val = dmin * f32(sm.y);

                for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 1u) {
                    let elem = sb * SUBBLOCK_SIZE + i;
                    let k = k_base + elem;
                    if (k < u.K) {
                        let a_val = A[row * u.K + k];
                        let q = get_q4(block.qs, elem);
                        let w = scale * f32(q) - min_val;
                        partial_sum = partial_sum + a_val * w;
                    }
                }
            }
        }
    }

    let sg_sum = subgroupAdd(partial_sum);
    let num_subgroups = (THREADS_PER_COL + sg_size - 1u) / sg_size;

    if (sg_id == 0u && local_id < THREADS_PER_COL) {
        let sg_idx = local_id / sg_size;
        batched_wg_sums[lid.y * MAX_SUBGROUPS_PER_ROW + sg_idx] = sg_sum;
    }

    workgroupBarrier();

    if (local_id == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < num_subgroups; i = i + 1u) {
            final_sum = final_sum + batched_wg_sums[lid.y * MAX_SUBGROUPS_PER_ROW + i];
        }
        C_f16[row * u.N + col] = f16(final_sum * u.alpha);
    }
}
