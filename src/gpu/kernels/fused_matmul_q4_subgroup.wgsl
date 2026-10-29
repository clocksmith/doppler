// AUTO-GENERATED from src/gpu/kernels/fused_matmul_q4_metal_simd16.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Q4_K GEMV with a configured subgroup partition per output row.

enable subgroups;

const QK_K: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 16u;
override THREADS_PER_COL_GEMV: u32 = 16u;
override USE_FULL_BLOCK_FAST_PATH: bool = false;

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

fn unpack_q4_word(word: u32, nibble_shift: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((word >> nibble_shift) & 0xFu),
        f32((word >> (nibble_shift + 8u)) & 0xFu),
        f32((word >> (nibble_shift + 16u)) & 0xFu),
        f32((word >> (nibble_shift + 24u)) & 0xFu)
    );
}

fn load_activations(base: u32, input_base: u32) -> vec4<f32> {
    if (USE_FULL_BLOCK_FAST_PATH || base + 3u < u.K) {
        return vec4<f32>(A[input_base + base], A[input_base + base + 1u], A[input_base + base + 2u], A[input_base + base + 3u]);
    }
    var result = vec4<f32>(0.0);
    if (base < u.K) { result.x = A[input_base + base]; }
    if (base + 1u < u.K) { result.y = A[input_base + base + 1u]; }
    if (base + 2u < u.K) { result.z = A[input_base + base + 2u]; }
    if (base + 3u < u.K) { result.w = A[input_base + base + 3u]; }
    return result;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) lane_id: u32,
    @builtin(subgroup_size) subgroup_size: u32
) {
    if (
        COLS_PER_WG * THREADS_PER_COL_GEMV != WORKGROUP_SIZE ||
        (THREADS_PER_COL_GEMV != 1u && THREADS_PER_COL_GEMV != 2u && THREADS_PER_COL_GEMV != 4u && THREADS_PER_COL_GEMV != 8u && THREADS_PER_COL_GEMV != 16u && THREADS_PER_COL_GEMV != 32u) ||
        subgroup_size < THREADS_PER_COL_GEMV ||
        subgroup_size % THREADS_PER_COL_GEMV != 0u
    ) {
        return;
    }

    let chunk_size = min(32u, QK_K / THREADS_PER_COL_GEMV);
    let lanes_per_subblock = max(1u, THREADS_PER_COL_GEMV / 8u);
    let subblocks_per_lane = max(1u, 8u / THREADS_PER_COL_GEMV);
    let rows_per_subgroup = subgroup_size / THREADS_PER_COL_GEMV;
    let subgroup_id = local_id.x / subgroup_size;
    let row_in_subgroup = lane_id / THREADS_PER_COL_GEMV;
    let lane_in_row = lane_id % THREADS_PER_COL_GEMV;
    let col = wg_id.x * COLS_PER_WG + subgroup_id * rows_per_subgroup + row_in_subgroup;
    let is_valid = col < u.N && wg_id.y < u.M;
    let input_base = wg_id.y * u.K;
    var partial_sum: f32 = 0.0;

    if (is_valid) {
        for (var block_idx: u32 = 0u; block_idx < u.num_blocks_per_row; block_idx = block_idx + 1u) {
            let storage_index = col * u.num_blocks_per_row + block_idx;
            let scales = B_q4k[storage_index].scales;
            let d_dmin = B_q4k[storage_index].d_dmin;
            for (var part = 0u; part < subblocks_per_lane; part = part + 1u) {
                let subblock = (lane_in_row / lanes_per_subblock) * subblocks_per_lane + part;
                let chunk_in_subblock = lane_in_row % lanes_per_subblock;
                let nibble_shift = (subblock & 1u) * 4u;
                let byte_offset = (subblock >> 1u) * 32u + chunk_in_subblock * chunk_size;
                let word_offset = byte_offset >> 2u;
                let scale_min = get_scale_min_k4(scales, subblock);
                let scale = unpack_f16_lo(d_dmin) * f32(scale_min.x);
                let min_value = unpack_f16_hi(d_dmin) * f32(scale_min.y);
                let activation_base = block_idx * QK_K + subblock * 32u + chunk_in_subblock * chunk_size;
                for (var word_idx = 0u; word_idx < chunk_size / 4u; word_idx = word_idx + 1u) {
                    let activations = load_activations(activation_base + word_idx * 4u, input_base);
                    let quantized = unpack_q4_word(B_q4k[storage_index].qs[word_offset + word_idx], nibble_shift);
                    partial_sum = partial_sum + dot(activations, scale * quantized - vec4<f32>(min_value));
                }
            }
        }
    }

    for (var offset = THREADS_PER_COL_GEMV / 2u; offset > 0u; offset = offset / 2u) {
        partial_sum = partial_sum + subgroupShuffleDown(partial_sum, offset);
    }

    if (lane_in_row == 0u && is_valid) {
        C[wg_id.y * u.N + col] = partial_sum * u.alpha;
    }
}
