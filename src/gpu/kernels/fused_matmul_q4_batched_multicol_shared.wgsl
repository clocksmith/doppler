// Fused Q4_K prefill matmul.
//
// Workgroup topology:
// - a small tile of rows (TILE_M)
// - multiple output columns per workgroup (COLS_PER_WG)
// - a few threads cooperate on each column (THREADS_PER_COL)
//
// This preserves Q4_K block traversal while loading each row's activation
// block once into workgroup memory for reuse across columns.

const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;

const MAX_TILE_M: u32 = 4u;
const MAX_COLS_PER_WG: u32 = 8u;
const MAX_THREADS_PER_COL: u32 = 4u;
const MAX_WORKGROUP_X: u32 = MAX_COLS_PER_WG * MAX_THREADS_PER_COL;
const MAX_SHARED_A: u32 = MAX_TILE_M * QK_K;
const MAX_PARTIALS: u32 = MAX_TILE_M * MAX_COLS_PER_WG * MAX_THREADS_PER_COL;

override WORKGROUP_X: u32 = 32u;
override TILE_M: u32 = 4u;
override COLS_PER_WG: u32 = 8u;
override THREADS_PER_COL: u32 = 4u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    num_blocks_per_row: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Q4KBlock {
    d_dmin: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B_q4k: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> shared_A: array<f32, MAX_SHARED_A>;
var<workgroup> partial_sums: array<f32, MAX_PARTIALS>;

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

@compute @workgroup_size(WORKGROUP_X, TILE_M, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if (
        TILE_M > MAX_TILE_M
        || COLS_PER_WG > MAX_COLS_PER_WG
        || THREADS_PER_COL > MAX_THREADS_PER_COL
        || WORKGROUP_X > MAX_WORKGROUP_X
        || COLS_PER_WG * THREADS_PER_COL != WORKGROUP_X
    ) {
        return;
    }

    let row_in_tile = lid.y;
    let row = wg_id.y * TILE_M + row_in_tile;
    let row_valid = row < u.M;
    let col_in_wg = lid.x / THREADS_PER_COL;
    let tid_in_col = lid.x % THREADS_PER_COL;
    let col = wg_id.x * COLS_PER_WG + col_in_wg;
    let col_valid = col < u.N;
    let is_valid = row_valid && col_valid;
    let row_shared_base = row_in_tile * QK_K;

    var partial_sum: f32 = 0.0;

    for (var block_idx: u32 = 0u; block_idx < u.num_blocks_per_row; block_idx = block_idx + 1u) {
        let block_k_base = block_idx * QK_K;

        for (var elem_idx: u32 = lid.x; elem_idx < QK_K; elem_idx = elem_idx + WORKGROUP_X) {
            let k = block_k_base + elem_idx;
            var a_value: f32 = 0.0;
            if (row_valid && k < u.K) {
                a_value = A[row * u.K + k];
            }
            shared_A[row_shared_base + elem_idx] = a_value;
        }
        workgroupBarrier();

        if (is_valid) {
            let block = B_q4k[col * u.num_blocks_per_row + block_idx];
            let d = unpack_f16_lo(block.d_dmin);
            let dmin = unpack_f16_hi(block.d_dmin);

            for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                let sm = get_scale_min_k4(block.scales, sb);
                let scale = d * f32(sm.x);
                let min_val = dmin * f32(sm.y);
                let sb_base = sb * SUBBLOCK_SIZE;

                for (var i: u32 = tid_in_col; i < SUBBLOCK_SIZE; i = i + THREADS_PER_COL) {
                    let elem = sb_base + i;
                    let a_value = shared_A[row_shared_base + elem];
                    let q = get_q4(block.qs, elem);
                    let w = scale * f32(q) - min_val;
                    partial_sum = partial_sum + a_value * w;
                }
            }
        }

        workgroupBarrier();
    }

    let partial_base = (row_in_tile * MAX_COLS_PER_WG + col_in_wg) * MAX_THREADS_PER_COL;
    partial_sums[partial_base + tid_in_col] = partial_sum;
    workgroupBarrier();

    if (tid_in_col == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < THREADS_PER_COL; i = i + 1u) {
            final_sum = final_sum + partial_sums[partial_base + i];
        }
        C[row * u.N + col] = final_sum * u.alpha;
    }
}
