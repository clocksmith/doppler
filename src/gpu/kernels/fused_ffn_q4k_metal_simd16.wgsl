// Fused Q4_K gate/up GEMV using 16 contiguous subgroup lanes per output row.

enable subgroups;

const QK_K: u32 = 256u;
const CHUNK_SIZE: u32 = 16u;
const LANES_PER_ROW: u32 = QK_K / CHUNK_SIZE;

override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 16u;
override THREADS_PER_COL: u32 = 16u;
override USE_FULL_BLOCK_FAST_PATH: bool = false;

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
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> W_gate_q4k: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read> W_up_q4k: array<Q4KBlock>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

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

fn load_activations(base: u32) -> vec4<f32> {
    if (USE_FULL_BLOCK_FAST_PATH || base + 3u < u.hidden_size) {
        return vec4<f32>(input[base], input[base + 1u], input[base + 2u], input[base + 3u]);
    }
    var result = vec4<f32>(0.0);
    if (base < u.hidden_size) { result.x = input[base]; }
    if (base + 1u < u.hidden_size) { result.y = input[base + 1u]; }
    if (base + 2u < u.hidden_size) { result.z = input[base + 2u]; }
    if (base + 3u < u.hidden_size) { result.w = input[base + 3u]; }
    return result;
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn gelu(x: f32) -> f32 {
    let c = 0.7978845608;
    let inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(clamp(inner, -15.0, 15.0)));
}

fn clamp_swiglu(x: f32) -> f32 {
    if (u.clamp_max <= 0.0 || u.activation != 0u) {
        return x;
    }
    return clamp(x, -u.clamp_max, u.clamp_max);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) lane_id: u32,
    @builtin(subgroup_size) subgroup_size: u32
) {
    if (
        COLS_PER_WG * THREADS_PER_COL != WORKGROUP_SIZE ||
        THREADS_PER_COL != LANES_PER_ROW ||
        subgroup_size < LANES_PER_ROW ||
        subgroup_size % LANES_PER_ROW != 0u
    ) {
        return;
    }

    let rows_per_subgroup = subgroup_size / LANES_PER_ROW;
    let subgroup_id = local_id.x / subgroup_size;
    let row_in_subgroup = lane_id / LANES_PER_ROW;
    let lane_in_row = lane_id % LANES_PER_ROW;
    let col = wg_id.x * COLS_PER_WG + subgroup_id * rows_per_subgroup + row_in_subgroup;
    let is_valid = col < u.intermediate_size;
    var partial_gate: f32 = 0.0;
    var partial_up: f32 = 0.0;

    if (is_valid) {
        let subblock = lane_in_row >> 1u;
        let half_chunk = lane_in_row & 1u;
        let nibble_shift = (subblock & 1u) * 4u;
        let byte_offset = (subblock >> 1u) * 32u + half_chunk * CHUNK_SIZE;
        let word_offset = byte_offset >> 2u;

        for (var block_idx: u32 = 0u; block_idx < u.num_blocks_per_row; block_idx = block_idx + 1u) {
            let block_offset = col * u.num_blocks_per_row + block_idx;
            let gate_block = W_gate_q4k[block_offset];
            let up_block = W_up_q4k[block_offset];
            let gate_scale_min = get_scale_min_k4(gate_block.scales, subblock);
            let up_scale_min = get_scale_min_k4(up_block.scales, subblock);
            let gate_scale = unpack_f16_lo(gate_block.d_dmin) * f32(gate_scale_min.x);
            let gate_min = unpack_f16_hi(gate_block.d_dmin) * f32(gate_scale_min.y);
            let up_scale = unpack_f16_lo(up_block.d_dmin) * f32(up_scale_min.x);
            let up_min = unpack_f16_hi(up_block.d_dmin) * f32(up_scale_min.y);
            let activation_base = block_idx * QK_K + lane_in_row * CHUNK_SIZE;

            for (var word_idx: u32 = 0u; word_idx < 4u; word_idx = word_idx + 1u) {
                let activations = load_activations(activation_base + word_idx * 4u);
                let gate_q = unpack_q4_word(gate_block.qs[word_offset + word_idx], nibble_shift);
                let up_q = unpack_q4_word(up_block.qs[word_offset + word_idx], nibble_shift);
                partial_gate = partial_gate + dot(
                    activations,
                    gate_scale * gate_q - vec4<f32>(gate_min)
                );
                partial_up = partial_up + dot(
                    activations,
                    up_scale * up_q - vec4<f32>(up_min)
                );
            }
        }
    }

    partial_gate = partial_gate + subgroupShuffleDown(partial_gate, 8u);
    partial_up = partial_up + subgroupShuffleDown(partial_up, 8u);
    partial_gate = partial_gate + subgroupShuffleDown(partial_gate, 4u);
    partial_up = partial_up + subgroupShuffleDown(partial_up, 4u);
    partial_gate = partial_gate + subgroupShuffleDown(partial_gate, 2u);
    partial_up = partial_up + subgroupShuffleDown(partial_up, 2u);
    partial_gate = partial_gate + subgroupShuffleDown(partial_gate, 1u);
    partial_up = partial_up + subgroupShuffleDown(partial_up, 1u);

    if (lane_in_row == 0u && is_valid) {
        var activated: f32;
        if (u.activation == 0u) {
            activated = silu(partial_gate);
        } else {
            activated = gelu(partial_gate);
        }
        output[col] = clamp_swiglu(activated * partial_up * u.alpha);
    }
}
