// Fused Q4_K Matmul Kernel - W4A16 (F16 activations, F16 output, GEMV)
//
// Computes C_f16 = A * dequant(B_q4k) for M=1 decode with f16 activations.

enable f16;
enable subgroups;

// Q4_K constants
const QK_K: u32 = 256u;
const BLOCK_SIZE: u32 = 144u;
const SUBBLOCK_SIZE: u32 = 32u;

override WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 256u;

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
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read> B_q4k: array<Q4KBlock>;
@group(0) @binding(4) var<storage, read_write> C_f16: array<f16>;

var<workgroup> wg_sums: array<f32, MAX_SUBGROUPS>;

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

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_f16a(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let col = wg_id.x;
    let local_id = lid.x;
    let is_valid = col < u.N;

    var partial_sum: f32 = 0.0;

    if (is_valid) {
        let num_blocks = u.num_blocks_per_row;
        let blocks_per_thread = (num_blocks + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
        let block_start = local_id * blocks_per_thread;
        let block_end = min(block_start + blocks_per_thread, num_blocks);

        for (var b: u32 = block_start; b < block_end; b = b + 1u) {
            let block = B_q4k[col * num_blocks + b];
            let d = unpack_f16_lo(block.d_dmin);
            let dmin = unpack_f16_hi(block.d_dmin);
            let k_base = b * QK_K;

            for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                let sm = get_scale_min_k4(block.scales, sb);
                let scale = d * f32(sm.x);
                let min_val = dmin * f32(sm.y);
                let sb_base = sb * SUBBLOCK_SIZE;

                for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                    let elem0 = sb_base + i;
                    let elem1 = elem0 + 1u;
                    let elem2 = elem0 + 2u;
                    let elem3 = elem0 + 3u;

                    let a0 = f32(A[k_base + elem0]);
                    let a1 = f32(A[k_base + elem1]);
                    let a2 = f32(A[k_base + elem2]);
                    let a3 = f32(A[k_base + elem3]);

                    let q0 = get_q4(block.qs, elem0);
                    let q1 = get_q4(block.qs, elem1);
                    let q2 = get_q4(block.qs, elem2);
                    let q3 = get_q4(block.qs, elem3);

                    let w0 = scale * f32(q0) - min_val;
                    let w1 = scale * f32(q1) - min_val;
                    let w2 = scale * f32(q2) - min_val;
                    let w3 = scale * f32(q3) - min_val;

                    partial_sum = partial_sum + a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
                }
            }
        }
    }

    let sg_sum = subgroupAdd(partial_sum);
    let subgroup_id = local_id / sg_size;
    let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;

    if (sg_id == 0u) {
        wg_sums[subgroup_id] = sg_sum;
    }

    workgroupBarrier();

    if (local_id == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < num_subgroups; i = i + 1u) {
            final_sum = final_sum + wg_sums[i];
        }
        C_f16[col] = f16(final_sum * u.alpha);
    }
}
