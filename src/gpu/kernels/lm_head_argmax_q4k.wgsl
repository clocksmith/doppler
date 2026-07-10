// lm-head-argmax-q4k.wgsl

override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 64u;
override THREADS_PER_COL: u32 = 4u;
override USE_FULL_BLOCK_FAST_PATH: bool = false;

const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;
const NUM_SUBBLOCKS: u32 = QK_K / SUBBLOCK_SIZE;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_COLS_PER_WG: u32 = 256u;
const NEG_INF: f32 = -3.402823e+38;

struct Uniforms {
    vocab_size: u32,
    hidden_size: u32,
    transpose_b: u32,
    workgroups_x: u32,
    pad_token_id: u32,
    logit_softcap: f32,
    output_index: u32,
    num_groups: u32,
}

struct Q4KBlock {
    d_dmin: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> hidden: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read_write> temp_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> temp_logits: array<f32>;

var<workgroup> partial_sums: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> candidate_values: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> candidate_indices: array<u32, MAX_WORKGROUP_SIZE>;

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

fn apply_softcap(x: f32, softcap: f32) -> f32 {
    if (softcap <= 0.0) {
        return x;
    }
    return softcap * tanh(x / softcap);
}

fn candidate_beats(candidate_value: f32, candidate_index: u32, best_value: f32, best_index: u32) -> bool {
    if (candidate_value > best_value) {
        return true;
    }
    if (candidate_value < best_value) {
        return false;
    }
    return candidate_index < best_index;
}

fn load_hidden(k: u32) -> f32 {
    if (k < u.hidden_size) {
        return hidden[k];
    }
    return 0.0;
}

fn accumulate_full_block(block: Q4KBlock, k_base: u32) -> f32 {
    let d = unpack_f16_lo(block.d_dmin);
    let dmin = unpack_f16_hi(block.d_dmin);
    var partial_sum: f32 = 0.0;

    for (var sb: u32 = 0u; sb < NUM_SUBBLOCKS; sb = sb + 1u) {
        let sb_base = sb * SUBBLOCK_SIZE;
        let sm = get_scale_min_k4(block.scales, sb);
        let scale = d * f32(sm.x);
        let min_val = dmin * f32(sm.y);
        let chunk = sb >> 1u;
        let nibble_shift = (sb & 1u) * 4u;
        let word_base = chunk * 8u;

        for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
            let k0 = k_base + sb_base + i;
            let word = block.qs[word_base + (i >> 2u)];
            let q0 = (word >> nibble_shift) & 0xFu;
            let q1 = (word >> (nibble_shift + 8u)) & 0xFu;
            let q2 = (word >> (nibble_shift + 16u)) & 0xFu;
            let q3 = (word >> (nibble_shift + 24u)) & 0xFu;

            let w0 = scale * f32(q0) - min_val;
            let w1 = scale * f32(q1) - min_val;
            let w2 = scale * f32(q2) - min_val;
            let w3 = scale * f32(q3) - min_val;

            partial_sum = partial_sum
                + hidden[k0] * w0
                + hidden[k0 + 1u] * w1
                + hidden[k0 + 2u] * w2
                + hidden[k0 + 3u] * w3;
        }
    }

    return partial_sum;
}

fn accumulate_partial_block(block: Q4KBlock, k_base: u32, remaining: u32) -> f32 {
    let d = unpack_f16_lo(block.d_dmin);
    let dmin = unpack_f16_hi(block.d_dmin);
    var partial_sum: f32 = 0.0;

    for (var sb: u32 = 0u; sb < NUM_SUBBLOCKS; sb = sb + 1u) {
        let sb_base = sb * SUBBLOCK_SIZE;
        if (sb_base >= remaining) {
            break;
        }

        let sm = get_scale_min_k4(block.scales, sb);
        let scale = d * f32(sm.x);
        let min_val = dmin * f32(sm.y);
        let chunk = sb >> 1u;
        let nibble_shift = (sb & 1u) * 4u;
        let word_base = chunk * 8u;

        for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
            let k0 = k_base + sb_base + i;
            let word = block.qs[word_base + (i >> 2u)];
            let q0 = (word >> nibble_shift) & 0xFu;
            let q1 = (word >> (nibble_shift + 8u)) & 0xFu;
            let q2 = (word >> (nibble_shift + 16u)) & 0xFu;
            let q3 = (word >> (nibble_shift + 24u)) & 0xFu;

            let w0 = scale * f32(q0) - min_val;
            let w1 = scale * f32(q1) - min_val;
            let w2 = scale * f32(q2) - min_val;
            let w3 = scale * f32(q3) - min_val;

            partial_sum = partial_sum
                + load_hidden(k0) * w0
                + load_hidden(k0 + 1u) * w1
                + load_hidden(k0 + 2u) * w2
                + load_hidden(k0 + 3u) * w3;
        }
    }

    return partial_sum;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn phase1(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let local_id = lid.x;
    let col_in_wg = local_id / THREADS_PER_COL;
    let thread_in_col = local_id % THREADS_PER_COL;
    let wg_linear = wg_id.y * u.workgroups_x + wg_id.x;
    let base_col = wg_linear * COLS_PER_WG;
    let col = base_col + col_in_wg;
    let is_valid = col < u.vocab_size && col != u.pad_token_id;

    let num_blocks = (u.hidden_size + QK_K - 1u) / QK_K;
    let has_full_blocks = USE_FULL_BLOCK_FAST_PATH && (u.hidden_size % QK_K) == 0u;
    var partial_sum: f32 = 0.0;

    if (is_valid) {
        for (var b: u32 = thread_in_col; b < num_blocks; b = b + THREADS_PER_COL) {
            let block = weights[col * num_blocks + b];
            let k_base = b * QK_K;
            if (has_full_blocks) {
                partial_sum = partial_sum + accumulate_full_block(block, k_base);
            } else {
                var remaining: u32 = 0u;
                if (k_base < u.hidden_size) {
                    remaining = min(QK_K, u.hidden_size - k_base);
                }
                partial_sum = partial_sum + accumulate_partial_block(block, k_base, remaining);
            }
        }
    }

    partial_sums[local_id] = partial_sum;
    workgroupBarrier();

    if (thread_in_col == 0u && col_in_wg < MAX_COLS_PER_WG) {
        let base = col_in_wg * THREADS_PER_COL;
        var sum = partial_sums[base];
        for (var i: u32 = 1u; i < THREADS_PER_COL; i = i + 1u) {
            sum = sum + partial_sums[base + i];
        }
        candidate_values[col_in_wg] = select(NEG_INF, apply_softcap(sum, u.logit_softcap), is_valid);
        candidate_indices[col_in_wg] = col;
    }
    workgroupBarrier();

    var stride = COLS_PER_WG / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            if (candidate_beats(
                candidate_values[local_id + stride],
                candidate_indices[local_id + stride],
                candidate_values[local_id],
                candidate_indices[local_id]
            )) {
                candidate_values[local_id] = candidate_values[local_id + stride];
                candidate_indices[local_id] = candidate_indices[local_id + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_id == 0u) {
        temp_logits[wg_linear] = candidate_values[0];
        temp_indices[wg_linear] = candidate_indices[0];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn phase2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local_id = lid.x;
    var local_best_value: f32 = NEG_INF;
    var local_best_index: u32 = 0u;

    var group = local_id;
    while (group < u.num_groups) {
        let candidate_value = temp_logits[group];
        let candidate_index = temp_indices[group];
        if (candidate_beats(candidate_value, candidate_index, local_best_value, local_best_index)) {
            local_best_value = candidate_value;
            local_best_index = candidate_index;
        }
        group = group + WORKGROUP_SIZE;
    }

    candidate_values[local_id] = local_best_value;
    candidate_indices[local_id] = local_best_index;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            if (candidate_beats(
                candidate_values[local_id + stride],
                candidate_indices[local_id + stride],
                candidate_values[local_id],
                candidate_indices[local_id]
            )) {
                candidate_values[local_id] = candidate_values[local_id + stride];
                candidate_indices[local_id] = candidate_indices[local_id + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_id == 0u) {
        output[u.output_index] = candidate_indices[0];
    }
}
