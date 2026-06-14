// gemma4_route_q4_matmul_f16a.wgsl

enable f16;
enable subgroups;

const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;
const THREADS_PER_COL_GEMV: u32 = 8u;

override TILE_M: u32 = 4u;
override THREADS_PER_COL: u32 = 64u;
const MAX_TILE_M: u32 = 4u;
const MAX_THREADS_PER_COL: u32 = 64u;
const MAX_SUBGROUPS_PER_ROW: u32 = 64u;

struct Uniforms {
    num_routes: u32,
    top_k: u32,
    N: u32,
    K: u32,
    num_blocks_per_row: u32,
    input_mode: u32,
    alpha: f32,
    _pad0: u32,
}

struct Q4KBlock {
    d_dmin: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read> route_indices: array<u32>;
@group(0) @binding(3) var<storage, read> B_q4k: array<Q4KBlock>;
@group(0) @binding(4) var<storage, read_write> C: array<f16>;

var<workgroup> route_wg_sums: array<f16, MAX_TILE_M * MAX_SUBGROUPS_PER_ROW>;

fn unpack_f16_lo(packed: u32) -> f16 {
    return f16(unpack2x16float(packed).x);
}

fn unpack_f16_hi(packed: u32) -> f16 {
    return f16(unpack2x16float(packed).y);
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

@compute @workgroup_size(THREADS_PER_COL, TILE_M, 1)
fn route_matmul_f16a(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if (TILE_M > MAX_TILE_M || THREADS_PER_COL > MAX_THREADS_PER_COL) {
        return;
    }

    let local_id = lid.x;
    let route_idx = wg_id.y * TILE_M + lid.y;
    let col = wg_id.x;
    let is_valid = route_idx < u.num_routes && col < u.N;
    let worker_id = local_id;

    var partial_sum: f16 = f16(0.0);

    if (is_valid && worker_id < THREADS_PER_COL_GEMV) {
        let expert_idx = route_indices[route_idx];
        let input_row = select(route_idx, route_idx / u.top_k, u.input_mode == 0u);
        let num_blocks = u.num_blocks_per_row;
        let tail_size = u.K & 255u;
        let full_blocks = num_blocks - select(0u, 1u, tail_size > 0u);

        for (var b: u32 = worker_id; b < full_blocks; b = b + THREADS_PER_COL_GEMV) {
            let block = B_q4k[(expert_idx * u.N + col) * num_blocks + b];
            let d = unpack_f16_lo(block.d_dmin);
            let dmin = unpack_f16_hi(block.d_dmin);
            let k_base = b * QK_K;

            for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                let sm = get_scale_min_k4(block.scales, sb);
                let scale = d * f16(sm.x);
                let min_val = dmin * f16(sm.y);
                let sb_base = sb * SUBBLOCK_SIZE;

                for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                    let elem0 = sb_base + i;
                    let k0 = k_base + elem0;

                    let a0 = A[input_row * u.K + k0];
                    let a1 = A[input_row * u.K + k0 + 1u];
                    let a2 = A[input_row * u.K + k0 + 2u];
                    let a3 = A[input_row * u.K + k0 + 3u];

                    let q0 = get_q4(block.qs, elem0);
                    let q1 = get_q4(block.qs, elem0 + 1u);
                    let q2 = get_q4(block.qs, elem0 + 2u);
                    let q3 = get_q4(block.qs, elem0 + 3u);

                    let w0 = scale * f16(q0) - min_val;
                    let w1 = scale * f16(q1) - min_val;
                    let w2 = scale * f16(q2) - min_val;
                    let w3 = scale * f16(q3) - min_val;

                    partial_sum = partial_sum + a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
                }
            }
        }

        if (tail_size > 0u) {
            let tail_block = full_blocks;
            if (tail_block % THREADS_PER_COL_GEMV == worker_id) {
                let block = B_q4k[(expert_idx * u.N + col) * num_blocks + tail_block];
                let d = unpack_f16_lo(block.d_dmin);
                let dmin = unpack_f16_hi(block.d_dmin);
                let k_base = tail_block * QK_K;

                for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                    let sb_base = sb * SUBBLOCK_SIZE;
                    if (sb_base >= tail_size) {
                        break;
                    }
                    let sm = get_scale_min_k4(block.scales, sb);
                    let scale = d * f16(sm.x);
                    let min_val = dmin * f16(sm.y);

                    for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                        let elem0 = sb_base + i;
                        let k0 = k_base + elem0;
                        let k1 = k0 + 1u;
                        let k2 = k0 + 2u;
                        let k3 = k0 + 3u;

                        var a0: f16 = f16(0.0);
                        var a1: f16 = f16(0.0);
                        var a2: f16 = f16(0.0);
                        var a3: f16 = f16(0.0);
                        if (k0 < u.K) { a0 = A[input_row * u.K + k0]; }
                        if (k1 < u.K) { a1 = A[input_row * u.K + k1]; }
                        if (k2 < u.K) { a2 = A[input_row * u.K + k2]; }
                        if (k3 < u.K) { a3 = A[input_row * u.K + k3]; }

                        let q0 = get_q4(block.qs, elem0);
                        let q1 = get_q4(block.qs, elem0 + 1u);
                        let q2 = get_q4(block.qs, elem0 + 2u);
                        let q3 = get_q4(block.qs, elem0 + 3u);

                        let w0 = scale * f16(q0) - min_val;
                        let w1 = scale * f16(q1) - min_val;
                        let w2 = scale * f16(q2) - min_val;
                        let w3 = scale * f16(q3) - min_val;

                        partial_sum = partial_sum + a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
                    }
                }
            }
        }
    }

    route_wg_sums[lid.y * MAX_SUBGROUPS_PER_ROW + local_id] = partial_sum;

    workgroupBarrier();

    if (local_id == 0u && is_valid) {
        var final_sum: f16 = f16(0.0);
        for (var i: u32 = 0u; i < THREADS_PER_COL_GEMV; i = i + 1u) {
            final_sum = final_sum + route_wg_sums[lid.y * MAX_SUBGROUPS_PER_ROW + i];
        }
        C[route_idx * u.N + col] = final_sum * f16(u.alpha);
    }
}
