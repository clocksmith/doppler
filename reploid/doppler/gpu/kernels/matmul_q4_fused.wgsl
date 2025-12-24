// Fused Q4_K Matmul Kernel - W4A16
// Directly computes C = A * dequant(B_q4k) without separate dequant pass
//
// For M=1 decode (GEMV): C[N] = A[K] * B_q4k^T[N,K]
// B_q4k is stored in Q4_K format: [N * ceil(K/256) * 144 bytes]
//
// Key optimizations:
// 1. Fused dequant + matmul - eliminates memory round-trip (2-3x speedup)
// 2. Subgroup operations for reduction
// 3. On-the-fly dequantization in registers
//
// A is f32 (activations), B_q4k is Q4_K quantized weights, C is f32.

enable f16;
enable subgroups;

// Q4_K constants
const QK_K: u32 = 256u;           // Elements per super-block
const BLOCK_SIZE: u32 = 144u;     // Bytes per Q4_K block
const SUBBLOCK_SIZE: u32 = 32u;   // Elements per sub-block

const WG_SIZE: u32 = 256u;

struct Uniforms {
    M: u32,           // Always 1 for GEMV
    N: u32,           // Output dimension
    K: u32,           // Inner dimension (must be multiple of 256 for Q4_K)
    alpha: f32,
    num_blocks_per_row: u32,  // K / 256
}

// Q4_K block structure (144 bytes)
// Layout: d(2) + dmin(2) + scales(12) + qs(128)
struct Q4KBlock {
    d_dmin: u32,          // d (f16) and dmin (f16) packed
    scales: array<u32, 3>, // 12 bytes of packed 6-bit scales
    qs: array<u32, 32>,   // 128 bytes of 4-bit quantized values
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B_q4k: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Shared memory for subgroup reduction
// Size 32 supports sg_size >= 8 (256 threads / 8 = 32 subgroups max)
var<workgroup> wg_sums: array<f32, 32>;

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

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let col = wg_id.x;  // Output column this workgroup computes
    let local_id = lid.x;

    // Track validity - NO early return to maintain uniform control flow for subgroupAdd
    let is_valid = col < uniforms.N;

    var partial_sum: f32 = 0.0;

    // Only do work if this column is valid
    if (is_valid) {
        // Each thread processes some Q4_K blocks
        let num_blocks = uniforms.num_blocks_per_row;
        let blocks_per_thread = (num_blocks + WG_SIZE - 1u) / WG_SIZE;
        let block_start = local_id * blocks_per_thread;
        let block_end = min(block_start + blocks_per_thread, num_blocks);

        // B_q4k layout: for row `col`, blocks are at col * num_blocks + block_idx
        let row_block_offset = col * num_blocks;

        for (var b: u32 = block_start; b < block_end; b = b + 1u) {
        let block = B_q4k[row_block_offset + b];

        // Extract super-block scale and min
        let d = unpack_f16_lo(block.d_dmin);
        let dmin = unpack_f16_hi(block.d_dmin);

        // Base element index for this block
        let k_base = b * QK_K;

        // Process all 256 elements in this block
        // Unroll by sub-block (8 sub-blocks of 32 elements each)
        for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
            let sm = get_scale_min_k4(block.scales, sb);
            let scale = d * f32(sm.x);
            let min_val = dmin * f32(sm.y);

            let sb_base = sb * SUBBLOCK_SIZE;

            // Process 32 elements in this sub-block
            // Unroll by 4 for better ILP
            for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                let elem0 = sb_base + i;
                let elem1 = sb_base + i + 1u;
                let elem2 = sb_base + i + 2u;
                let elem3 = sb_base + i + 3u;

                let k0 = k_base + elem0;
                let k1 = k_base + elem1;
                let k2 = k_base + elem2;
                let k3 = k_base + elem3;

                // Load activations
                let a0 = A[k0];
                let a1 = A[k1];
                let a2 = A[k2];
                let a3 = A[k3];

                // Dequantize weights on-the-fly
                let q0 = get_q4(block.qs, elem0);
                let q1 = get_q4(block.qs, elem1);
                let q2 = get_q4(block.qs, elem2);
                let q3 = get_q4(block.qs, elem3);

                let w0 = scale * f32(q0) - min_val;
                let w1 = scale * f32(q1) - min_val;
                let w2 = scale * f32(q2) - min_val;
                let w3 = scale * f32(q3) - min_val;

                // Accumulate
                partial_sum = partial_sum + a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
            }
        }
        }
    }  // end if (is_valid)

    // Subgroup reduction - ALL threads must execute (uniform control flow)
    let sg_sum = subgroupAdd(partial_sum);

    // Inter-subgroup reduction via shared memory
    let subgroup_id = local_id / sg_size;
    let num_subgroups = (WG_SIZE + sg_size - 1u) / sg_size;

    if (sg_id == 0u) {
        wg_sums[subgroup_id] = sg_sum;
    }

    workgroupBarrier();

    // Thread 0 does final reduction and writes result (only if valid column)
    if (local_id == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < num_subgroups; i = i + 1u) {
            final_sum = final_sum + wg_sums[i];
        }
        C[col] = final_sum * uniforms.alpha;
    }
}

// ============================================================================
// Multi-column GEMV for large vocab (LM head)
// ============================================================================
// For large N (e.g., vocab=262144), single-column-per-workgroup is inefficient:
// - 262K workgroups with only ~5 blocks per thread
// - GPU launch overhead dominates
//
// This variant processes COLS_PER_WG columns per workgroup:
// - 262144/32 = 8192 workgroups (32x fewer)
// - Each thread processes multiple columns, amortizing A loads
// - Much better GPU occupancy and throughput
//
// Workgroup layout: 256 threads = 8 threads per column × 32 columns
const COLS_PER_WG: u32 = 32u;
const THREADS_PER_COL_GEMV: u32 = 8u;  // 256 / 32 = 8

// Shared memory for reduction: 32 columns × 8 max subgroups = 256
var<workgroup> multicol_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main_multicol(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let local_id = lid.x;

    // Which column within this workgroup (0..31)
    let col_in_wg = local_id / THREADS_PER_COL_GEMV;
    // Which thread within the column (0..7)
    let tid_in_col = local_id % THREADS_PER_COL_GEMV;

    // Global column index
    let col = wg_id.x * COLS_PER_WG + col_in_wg;

    // Track validity
    let is_valid = col < uniforms.N;

    var partial_sum: f32 = 0.0;

    if (is_valid) {
        let num_blocks = uniforms.num_blocks_per_row;
        let row_block_offset = col * num_blocks;

        // Each of the 8 threads processes every 8th block
        for (var b: u32 = tid_in_col; b < num_blocks; b = b + THREADS_PER_COL_GEMV) {
            let block = B_q4k[row_block_offset + b];
            let d = unpack_f16_lo(block.d_dmin);
            let dmin = unpack_f16_hi(block.d_dmin);
            let k_base = b * QK_K;

            // Process all 8 sub-blocks
            for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                let sm = get_scale_min_k4(block.scales, sb);
                let scale = d * f32(sm.x);
                let min_val = dmin * f32(sm.y);
                let sb_base = sb * SUBBLOCK_SIZE;

                // Unroll by 4 for better ILP
                for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                    let k0 = k_base + sb_base + i;
                    let k1 = k0 + 1u;
                    let k2 = k0 + 2u;
                    let k3 = k0 + 3u;

                    let a0 = A[k0];
                    let a1 = A[k1];
                    let a2 = A[k2];
                    let a3 = A[k3];

                    let q0 = get_q4(block.qs, sb_base + i);
                    let q1 = get_q4(block.qs, sb_base + i + 1u);
                    let q2 = get_q4(block.qs, sb_base + i + 2u);
                    let q3 = get_q4(block.qs, sb_base + i + 3u);

                    let w0 = scale * f32(q0) - min_val;
                    let w1 = scale * f32(q1) - min_val;
                    let w2 = scale * f32(q2) - min_val;
                    let w3 = scale * f32(q3) - min_val;

                    partial_sum = partial_sum + a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
                }
            }
        }
    }

    // Reduction within each column's 8 threads
    // Use shared memory since threads for one column may span multiple subgroups
    multicol_sums[local_id] = partial_sum;
    workgroupBarrier();

    // Thread 0 of each column reduces its 8 values
    if (tid_in_col == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        let base = col_in_wg * THREADS_PER_COL_GEMV;
        for (var i: u32 = 0u; i < THREADS_PER_COL_GEMV; i = i + 1u) {
            final_sum = final_sum + multicol_sums[base + i];
        }
        C[col] = final_sum * uniforms.alpha;
    }
}

// ============================================================================
// Batched version for prefill (M > 1)
// Uses 2D dispatch: workgroup (x,y) computes output C[y*TILE_M : (y+1)*TILE_M, x]
// Each workgroup computes TILE_M rows × 1 column
//
// IMPORTANT: Previous version used 16 threads per column which caused subgroup
// mixing when sg_size=32 (two columns in same subgroup). This version uses
// 64 threads per column to ensure correct subgroup reduction.
const TILE_M: u32 = 4u;
const THREADS_PER_COL: u32 = 64u;
const MAX_SUBGROUPS_PER_ROW: u32 = 16u;  // Support sg_size >= 4 (64/4 = 16)

// Shared memory for per-row subgroup reduction: 4 rows × 16 max subgroups = 64
var<workgroup> batched_wg_sums: array<f32, 64>;

@compute @workgroup_size(64, 4, 1)
fn main_batched(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let local_id = lid.x;
    let row = wg_id.y * TILE_M + lid.y;
    let col = wg_id.x;  // One column per workgroup X (no more /16 mixing)

    // Track validity - NO early return to maintain uniform control flow for subgroupAdd
    let is_valid = row < uniforms.M && col < uniforms.N;

    var partial_sum: f32 = 0.0;

    // Only do work if this output cell is valid
    if (is_valid) {
        let num_blocks = uniforms.num_blocks_per_row;
        let row_block_offset = col * num_blocks;

        // Each thread processes every 64th block (instead of 16th)
        for (var b: u32 = local_id; b < num_blocks; b = b + THREADS_PER_COL) {
            let block = B_q4k[row_block_offset + b];
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
                    if (k < uniforms.K) {
                        let a_val = A[row * uniforms.K + k];
                        let q = get_q4(block.qs, elem);
                        let w = scale * f32(q) - min_val;
                        partial_sum = partial_sum + a_val * w;
                    }
                }
            }
        }
    }  // end if (is_valid)

    // Subgroup reduction - ALL threads must execute (uniform control flow)
    let sg_sum = subgroupAdd(partial_sum);

    // Store subgroup results to shared memory (per-row)
    let num_subgroups = (THREADS_PER_COL + sg_size - 1u) / sg_size;

    if (sg_id == 0u && local_id < THREADS_PER_COL) {
        let sg_idx = local_id / sg_size;
        batched_wg_sums[lid.y * MAX_SUBGROUPS_PER_ROW + sg_idx] = sg_sum;
    }

    workgroupBarrier();

    // Thread 0 of each row does final reduction and writes result
    if (local_id == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < num_subgroups; i = i + 1u) {
            final_sum = final_sum + batched_wg_sums[lid.y * MAX_SUBGROUPS_PER_ROW + i];
        }
        C[row * uniforms.N + col] = final_sum * uniforms.alpha;
    }
}
