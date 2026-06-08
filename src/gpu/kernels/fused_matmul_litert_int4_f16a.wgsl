enable f16;

override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 8u;
override MULTICOL_COLS_PER_WG: u32 = 8u;
override MULTICOL_THREADS_PER_COL: u32 = 32u;
override WEIGHT_SCALE: f32 = 0.0625;
override STORAGE_OFFSET_BINARY: u32 = 0u;

const MAX_WORKGROUP_SIZE: u32 = 256u;

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
@group(0) @binding(3) var<storage, read_write> C_f16: array<f16>;

var<workgroup> partials: array<f32, MAX_WORKGROUP_SIZE>;

fn packed_byte(byte_index: u32) -> u32 {
    let word = B_words[byte_index >> 2u];
    let shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn signed_int4(byte_value: u32, high_nibble: bool) -> f32 {
    let value = select(byte_value & 0x0fu, (byte_value >> 4u) & 0x0fu, high_nibble);
    if (STORAGE_OFFSET_BINARY == 1u) {
        return f32(value) - 8.0;
    }
    return select(f32(value), f32(i32(value) - 16), value >= 8u);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_multicol_f16a(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let local_id = lid.x;
    let lanes_per_col = WORKGROUP_SIZE / MULTICOL_COLS_PER_WG;
    let col_in_wg = local_id / lanes_per_col;
    let lane_in_col = local_id - col_in_wg * lanes_per_col;
    let row = wg_id.y;
    let col = wg_id.x * MULTICOL_COLS_PER_WG + col_in_wg;
    let packed_bytes_per_row = (u.K + 1u) >> 1u;

    var sum: f32 = 0.0;
    if (row < u.M && col < u.N) {
        let row_base = col * packed_bytes_per_row;
        for (var k: u32 = lane_in_col; k < u.K; k = k + lanes_per_col) {
            let byte_value = packed_byte(row_base + (k >> 1u));
            let weight = signed_int4(byte_value, (k & 1u) == 1u) * WEIGHT_SCALE;
            sum = sum + f32(A[row * u.K + k]) * weight;
        }
    }

    partials[local_id] = sum;
    workgroupBarrier();

    if (lane_in_col == 0u && row < u.M && col < u.N) {
        var final_sum: f32 = 0.0;
        let base = col_in_wg * lanes_per_col;
        for (var lane: u32 = 0u; lane < lanes_per_col; lane = lane + 1u) {
            final_sum = final_sum + partials[base + lane];
        }
        C_f16[row * u.N + col] = f16(final_sum * u.alpha);
    }
}
