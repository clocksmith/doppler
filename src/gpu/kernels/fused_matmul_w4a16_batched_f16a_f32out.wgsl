enable f16;

override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 8u;
override LANES_PER_COL: u32 = 32u;
override TILE_M: u32 = 4u;
override SCALE_DTYPE: u32 = 1u;

const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_TILE_M: u32 = 4u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transpose_b: u32,
    workgroups_x: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read> B_words: array<u32>;
@group(0) @binding(3) var<storage, read> S_words: array<u32>;
@group(0) @binding(4) var<storage, read_write> C_f32: array<f32>;

var<workgroup> partials: array<f32, MAX_WORKGROUP_SIZE * MAX_TILE_M>;

fn packed_byte(byte_index: u32) -> u32 {
    let word = B_words[byte_index >> 2u];
    let shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn packed_scale_half(scale_index: u32) -> u32 {
    let word = S_words[scale_index >> 1u];
    return select(word & 0xffffu, (word >> 16u) & 0xffffu, (scale_index & 1u) == 1u);
}

fn scale_value(scale_index: u32) -> f32 {
    if (SCALE_DTYPE == 2u) {
        return bitcast<f32>(S_words[scale_index]);
    }
    let half = packed_scale_half(scale_index);
    if (SCALE_DTYPE == 1u) {
        return bitcast<f32>(half << 16u);
    }
    let pair = unpack2x16float(S_words[scale_index >> 1u]);
    return select(pair.x, pair.y, (scale_index & 1u) == 1u);
}

fn offset_binary_int4(byte_value: u32, high_nibble: bool) -> f32 {
    let value = select(byte_value & 0x0fu, (byte_value >> 4u) & 0x0fu, high_nibble);
    return f32(value) - 8.0;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_batched_f16a_f32out(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if (TILE_M > MAX_TILE_M) {
        return;
    }

    let local_id = lid.x;
    let col_in_wg = local_id / LANES_PER_COL;
    let lane_in_col = local_id - col_in_wg * LANES_PER_COL;
    let col = wg_id.x * COLS_PER_WG + col_in_wg;
    let row_base_out = wg_id.y * TILE_M;
    let groups_per_row = (u.K + 31u) >> 5u;

    var sums: array<f32, MAX_TILE_M>;
    for (var row_in_tile: u32 = 0u; row_in_tile < TILE_M; row_in_tile = row_in_tile + 1u) {
        sums[row_in_tile] = 0.0;
    }

    if (col < u.N) {
        let weight_base = col * groups_per_row * 16u;
        let scale_base = col * groups_per_row;
        for (var k: u32 = lane_in_col; k < u.K; k = k + LANES_PER_COL) {
            let group = k >> 5u;
            let lane = k & 31u;
            let byte_value = packed_byte(weight_base + group * 16u + (lane >> 1u));
            let weight = offset_binary_int4(byte_value, (lane & 1u) == 1u) * scale_value(scale_base + group);
            for (var row_in_tile: u32 = 0u; row_in_tile < TILE_M; row_in_tile = row_in_tile + 1u) {
                let row = row_base_out + row_in_tile;
                if (row < u.M) {
                    sums[row_in_tile] = sums[row_in_tile] + f32(A[row * u.K + k]) * weight;
                }
            }
        }
    }

    for (var row_in_tile: u32 = 0u; row_in_tile < TILE_M; row_in_tile = row_in_tile + 1u) {
        partials[row_in_tile * WORKGROUP_SIZE + local_id] = sums[row_in_tile];
    }
    workgroupBarrier();

    if (lane_in_col == 0u && col < u.N) {
        let lane_base = col_in_wg * LANES_PER_COL;
        for (var row_in_tile: u32 = 0u; row_in_tile < TILE_M; row_in_tile = row_in_tile + 1u) {
            let row = row_base_out + row_in_tile;
            if (row < u.M) {
                var final_sum: f32 = 0.0;
                let partial_base = row_in_tile * WORKGROUP_SIZE + lane_base;
                for (var lane: u32 = 0u; lane < LANES_PER_COL; lane = lane + 1u) {
                    final_sum = final_sum + partials[partial_base + lane];
                }
                C_f32[row * u.N + col] = final_sum * u.alpha;
            }
        }
    }
}
