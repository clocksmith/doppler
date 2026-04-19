// Fused FFN Q4_K Kernel - Native F16 Activation Reads
//
// Fuses gate + up weight projections for Q4_K quantized weights while reading
// f16 activations directly from storage. Accumulation and output stay f32 so
// the downstream FFN contract remains unchanged.

enable f16;
enable subgroups;

const QK_K: u32 = 256u;
const BLOCK_SIZE: u32 = 144u;
const SUBBLOCK_SIZE: u32 = 32u;

const MAX_WORKGROUP_SIZE: u32 = 256u;
override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 32u;
override THREADS_PER_COL: u32 = 8u;

struct Uniforms {
    M: u32,
    hidden_size: u32,
    intermediate_size: u32,
    alpha: f32,
    activation: u32,
    num_blocks_per_row: u32,
    clamp_max: f32,
    _pad0: u32,
}

struct Q4KBlock {
    d_dmin: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> W_gate_q4k: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read> W_up_q4k: array<Q4KBlock>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> multicol_gate: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> multicol_up: array<f32, MAX_WORKGROUP_SIZE>;

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

fn clamp_swiglu(x: f32) -> f32 {
    if (u.clamp_max <= 0.0 || u.activation != 0u) {
        return x;
    }
    return clamp(x, -u.clamp_max, u.clamp_max);
}

fn unpack_q4_word(word: u32, nibble_shift: u32) -> vec4<u32> {
    return vec4<u32>(
        (word >> nibble_shift) & 0xFu,
        (word >> (8u + nibble_shift)) & 0xFu,
        (word >> (16u + nibble_shift)) & 0xFu,
        (word >> (24u + nibble_shift)) & 0xFu
    );
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn gelu(x: f32) -> f32 {
    let c = 0.7978845608;
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || COLS_PER_WG * THREADS_PER_COL != WORKGROUP_SIZE) {
        return;
    }

    let local_id = lid.x;
    let col_in_wg = local_id / THREADS_PER_COL;
    let tid_in_col = local_id % THREADS_PER_COL;
    let col = wg_id.x * COLS_PER_WG + col_in_wg;
    let is_valid = col < u.intermediate_size;

    var partial_gate: f32 = 0.0;
    var partial_up: f32 = 0.0;

    if (is_valid) {
        let num_blocks = u.num_blocks_per_row;

        for (var b: u32 = tid_in_col; b < num_blocks; b = b + THREADS_PER_COL) {
            let gate_block = W_gate_q4k[col * num_blocks + b];
            let up_block = W_up_q4k[col * num_blocks + b];

            let gate_d = unpack_f16_lo(gate_block.d_dmin);
            let gate_dmin = unpack_f16_hi(gate_block.d_dmin);
            let up_d = unpack_f16_lo(up_block.d_dmin);
            let up_dmin = unpack_f16_hi(up_block.d_dmin);

            let k_base = b * QK_K;

            for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                let sb_base = sb * SUBBLOCK_SIZE;
                if (k_base + sb_base >= u.hidden_size) { break; }

                let gate_sm = get_scale_min_k4(gate_block.scales, sb);
                let gate_scale = gate_d * f32(gate_sm.x);
                let gate_min = gate_dmin * f32(gate_sm.y);

                let up_sm = get_scale_min_k4(up_block.scales, sb);
                let up_scale = up_d * f32(up_sm.x);
                let up_min = up_dmin * f32(up_sm.y);

                let chunk = sb >> 1u;
                let nibble_shift = (sb & 1u) * 4u;
                let word_base = chunk * 8u;

                for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                    let k0 = k_base + sb_base + i;
                    let k1 = k0 + 1u;
                    let k2 = k0 + 2u;
                    let k3 = k0 + 3u;

                    let a0 = f32(input[k0]);
                    let a1 = f32(input[k1]);
                    let a2 = f32(input[k2]);
                    let a3 = f32(input[k3]);

                    let word_idx = word_base + (i >> 2u);
                    let gate_q = unpack_q4_word(gate_block.qs[word_idx], nibble_shift);
                    let up_q = unpack_q4_word(up_block.qs[word_idx], nibble_shift);

                    let gw0 = gate_scale * f32(gate_q.x) - gate_min;
                    let gw1 = gate_scale * f32(gate_q.y) - gate_min;
                    let gw2 = gate_scale * f32(gate_q.z) - gate_min;
                    let gw3 = gate_scale * f32(gate_q.w) - gate_min;

                    let uw0 = up_scale * f32(up_q.x) - up_min;
                    let uw1 = up_scale * f32(up_q.y) - up_min;
                    let uw2 = up_scale * f32(up_q.z) - up_min;
                    let uw3 = up_scale * f32(up_q.w) - up_min;

                    partial_gate = partial_gate + a0 * gw0 + a1 * gw1 + a2 * gw2 + a3 * gw3;
                    partial_up = partial_up + a0 * uw0 + a1 * uw1 + a2 * uw2 + a3 * uw3;
                }
            }
        }
    }

    multicol_gate[local_id] = partial_gate;
    multicol_up[local_id] = partial_up;
    workgroupBarrier();

    if (tid_in_col == 0u && is_valid) {
        var final_gate: f32 = 0.0;
        var final_up: f32 = 0.0;
        let base = col_in_wg * THREADS_PER_COL;
        for (var i: u32 = 0u; i < THREADS_PER_COL; i = i + 1u) {
            final_gate = final_gate + multicol_gate[base + i];
            final_up = final_up + multicol_up[base + i];
        }

        var activated: f32;
        if (u.activation == 0u) {
            activated = silu(final_gate);
        } else {
            activated = gelu(final_gate);
        }

        output[col] = clamp_swiglu(activated * final_up * u.alpha);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_batched(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || COLS_PER_WG * THREADS_PER_COL != WORKGROUP_SIZE) {
        return;
    }

    let local_id = lid.x;
    let batch_idx = wg_id.y;
    let col_in_wg = local_id / THREADS_PER_COL;
    let tid_in_col = local_id % THREADS_PER_COL;
    let col = wg_id.x * COLS_PER_WG + col_in_wg;
    let is_valid = col < u.intermediate_size && batch_idx < u.M;

    var partial_gate: f32 = 0.0;
    var partial_up: f32 = 0.0;

    if (is_valid) {
        let num_blocks = u.num_blocks_per_row;
        let input_base = batch_idx * u.hidden_size;

        for (var b: u32 = tid_in_col; b < num_blocks; b = b + THREADS_PER_COL) {
            let gate_block = W_gate_q4k[col * num_blocks + b];
            let up_block = W_up_q4k[col * num_blocks + b];

            let gate_d = unpack_f16_lo(gate_block.d_dmin);
            let gate_dmin = unpack_f16_hi(gate_block.d_dmin);
            let up_d = unpack_f16_lo(up_block.d_dmin);
            let up_dmin = unpack_f16_hi(up_block.d_dmin);

            let k_base = b * QK_K;

            for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                let sb_base = sb * SUBBLOCK_SIZE;
                if (k_base + sb_base >= u.hidden_size) { break; }

                let gate_sm = get_scale_min_k4(gate_block.scales, sb);
                let gate_scale = gate_d * f32(gate_sm.x);
                let gate_min = gate_dmin * f32(gate_sm.y);

                let up_sm = get_scale_min_k4(up_block.scales, sb);
                let up_scale = up_d * f32(up_sm.x);
                let up_min = up_dmin * f32(up_sm.y);

                let chunk = sb >> 1u;
                let nibble_shift = (sb & 1u) * 4u;
                let word_base = chunk * 8u;

                for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                    let k0 = k_base + sb_base + i;

                    let a0 = f32(input[input_base + k0]);
                    let a1 = f32(input[input_base + k0 + 1u]);
                    let a2 = f32(input[input_base + k0 + 2u]);
                    let a3 = f32(input[input_base + k0 + 3u]);

                    let word_idx = word_base + (i >> 2u);
                    let gate_q = unpack_q4_word(gate_block.qs[word_idx], nibble_shift);
                    let up_q = unpack_q4_word(up_block.qs[word_idx], nibble_shift);

                    let gw0 = gate_scale * f32(gate_q.x) - gate_min;
                    let gw1 = gate_scale * f32(gate_q.y) - gate_min;
                    let gw2 = gate_scale * f32(gate_q.z) - gate_min;
                    let gw3 = gate_scale * f32(gate_q.w) - gate_min;

                    let uw0 = up_scale * f32(up_q.x) - up_min;
                    let uw1 = up_scale * f32(up_q.y) - up_min;
                    let uw2 = up_scale * f32(up_q.z) - up_min;
                    let uw3 = up_scale * f32(up_q.w) - up_min;

                    partial_gate = partial_gate + a0 * gw0 + a1 * gw1 + a2 * gw2 + a3 * gw3;
                    partial_up = partial_up + a0 * uw0 + a1 * uw1 + a2 * uw2 + a3 * uw3;
                }
            }
        }
    }

    multicol_gate[local_id] = partial_gate;
    multicol_up[local_id] = partial_up;
    workgroupBarrier();

    if (tid_in_col == 0u && is_valid) {
        var final_gate: f32 = 0.0;
        var final_up: f32 = 0.0;
        let base = col_in_wg * THREADS_PER_COL;
        for (var i: u32 = 0u; i < THREADS_PER_COL; i = i + 1u) {
            final_gate = final_gate + multicol_gate[base + i];
            final_up = final_up + multicol_up[base + i];
        }

        var activated: f32;
        if (u.activation == 0u) {
            activated = silu(final_gate);
        } else {
            activated = gelu(final_gate);
        }

        let out_offset = batch_idx * u.intermediate_size + col;
        output[out_offset] = clamp_swiglu(activated * final_up * u.alpha);
    }
}
