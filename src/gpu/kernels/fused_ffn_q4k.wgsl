// Fused FFN Q4_K Kernel - Multi-Column Optimized
//
// Fuses gate + up weight projections for Q4_K quantized weights.
// Dequantizes on-the-fly and applies activation (SiLU/GeGLU).
//
// Key optimizations:
// 1. Multi-column: 32 output columns per workgroup (32x fewer dispatches)
// 2. 8 threads per column share reduction work
// 3. Processes BOTH gate and up weights simultaneously
// 4. Input loaded once from global, used for both projections
//
// Memory savings vs separate kernels:
// - Single input read instead of 2 separate matmuls
// - No intermediate gate/up buffers
// - Fused activation eliminates memory round-trip
//
// Based on research from:
// - llama.cpp Metal: N_R0_Q4_K=2 rows/simdgroup, simdgroup-based reduction
// - WebLLM/MLC-LLM: Uses TVM for kernel generation with tiling optimizations

enable subgroups;

// Q4_K constants
const QK_K: u32 = 256u;           // Elements per super-block
const BLOCK_SIZE: u32 = 144u;     // Bytes per Q4_K block
const SUBBLOCK_SIZE: u32 = 32u;   // Elements per sub-block

// Multi-column layout: 256 threads = 8 threads/col × 32 columns
// These can be overridden via pipeline constants from config
const MAX_WORKGROUP_SIZE: u32 = 256u;
override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 32u;
override THREADS_PER_COL: u32 = 8u;  // 256 / 32 = 8

struct Uniforms {
    M: u32,                   // Batch size (usually 1 for decode)
    hidden_size: u32,         // Input dimension (K, must be multiple of 256)
    intermediate_size: u32,   // Output dimension (N)
    alpha: f32,               // Scale factor
    activation: u32,          // 0=silu, 1=gelu
    num_blocks_per_row: u32,  // hidden_size / 256
    clamp_max: f32,           // SwiGLU clamp (0 = disabled)
    _pad0: u32,
}

// Q4_K block structure (144 bytes)
struct Q4KBlock {
    d_dmin: u32,          // d (f16) and dmin (f16) packed
    scales: array<u32, 3>, // 12 bytes of packed 6-bit scales
    qs: array<u32, 32>,   // 128 bytes of 4-bit quantized values
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> W_gate_q4k: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read> W_up_q4k: array<Q4KBlock>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared memory for reduction: 32 columns × 8 threads × 2 (gate+up)
var<workgroup> multicol_gate: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> multicol_up: array<f32, MAX_WORKGROUP_SIZE>;

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

fn clamp_swiglu(x: f32) -> f32 {
    if (u.clamp_max <= 0.0 || u.activation != 0u) {
        return x;
    }
    return clamp(x, -u.clamp_max, u.clamp_max);
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

// SiLU activation: x * sigmoid(x)
fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// GELU activation (approximate)
fn gelu(x: f32) -> f32 {
    let c = 0.7978845608; // sqrt(2/pi)
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

// Main entry point: multi-column decode (M=1)
// Each workgroup computes 32 output columns
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

    // Which column within this workgroup (0..31)
    let col_in_wg = local_id / THREADS_PER_COL;
    // Which thread within the column (0..7)
    let tid_in_col = local_id % THREADS_PER_COL;

    // Global column index
    let col = wg_id.x * COLS_PER_WG + col_in_wg;

    // Track validity
    let is_valid = col < u.intermediate_size;

    var partial_gate: f32 = 0.0;
    var partial_up: f32 = 0.0;

    if (is_valid) {
        let num_blocks = u.num_blocks_per_row;

        // Each of the 8 threads processes every 8th block
        for (var b: u32 = tid_in_col; b < num_blocks; b = b + THREADS_PER_COL) {
            let gate_block = W_gate_q4k[col * num_blocks + b];
            let up_block = W_up_q4k[col * num_blocks + b];

            let gate_d = unpack_f16_lo(gate_block.d_dmin);
            let gate_dmin = unpack_f16_hi(gate_block.d_dmin);
            let up_d = unpack_f16_lo(up_block.d_dmin);
            let up_dmin = unpack_f16_hi(up_block.d_dmin);

            let k_base = b * QK_K;

            // Process all 8 sub-blocks
            for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
                let gate_sm = get_scale_min_k4(gate_block.scales, sb);
                let gate_scale = gate_d * f32(gate_sm.x);
                let gate_min = gate_dmin * f32(gate_sm.y);

                let up_sm = get_scale_min_k4(up_block.scales, sb);
                let up_scale = up_d * f32(up_sm.x);
                let up_min = up_dmin * f32(up_sm.y);

                let sb_base = sb * SUBBLOCK_SIZE;

                // Unroll by 4 for better ILP
                for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                    let k0 = k_base + sb_base + i;
                    let k1 = k0 + 1u;
                    let k2 = k0 + 2u;
                    let k3 = k0 + 3u;

                    // Load activations (shared across gate and up)
                    let a0 = input[k0];
                    let a1 = input[k1];
                    let a2 = input[k2];
                    let a3 = input[k3];

                    // Gate weights
                    let gq0 = get_q4(gate_block.qs, sb_base + i);
                    let gq1 = get_q4(gate_block.qs, sb_base + i + 1u);
                    let gq2 = get_q4(gate_block.qs, sb_base + i + 2u);
                    let gq3 = get_q4(gate_block.qs, sb_base + i + 3u);

                    let gw0 = gate_scale * f32(gq0) - gate_min;
                    let gw1 = gate_scale * f32(gq1) - gate_min;
                    let gw2 = gate_scale * f32(gq2) - gate_min;
                    let gw3 = gate_scale * f32(gq3) - gate_min;

                    // Up weights
                    let uq0 = get_q4(up_block.qs, sb_base + i);
                    let uq1 = get_q4(up_block.qs, sb_base + i + 1u);
                    let uq2 = get_q4(up_block.qs, sb_base + i + 2u);
                    let uq3 = get_q4(up_block.qs, sb_base + i + 3u);

                    let uw0 = up_scale * f32(uq0) - up_min;
                    let uw1 = up_scale * f32(uq1) - up_min;
                    let uw2 = up_scale * f32(uq2) - up_min;
                    let uw3 = up_scale * f32(uq3) - up_min;

                    // Accumulate both projections
                    partial_gate = partial_gate + a0 * gw0 + a1 * gw1 + a2 * gw2 + a3 * gw3;
                    partial_up = partial_up + a0 * uw0 + a1 * uw1 + a2 * uw2 + a3 * uw3;
                }
            }
        }
    }

    // Reduction within each column's 8 threads via shared memory
    // (threads for one column may span multiple subgroups on some GPUs)
    multicol_gate[local_id] = partial_gate;
    multicol_up[local_id] = partial_up;
    workgroupBarrier();

    // Thread 0 of each column reduces its 8 values
    if (tid_in_col == 0u && is_valid) {
        var final_gate: f32 = 0.0;
        var final_up: f32 = 0.0;
        let base = col_in_wg * THREADS_PER_COL;
        for (var i: u32 = 0u; i < THREADS_PER_COL; i = i + 1u) {
            final_gate = final_gate + multicol_gate[base + i];
            final_up = final_up + multicol_up[base + i];
        }

        // Apply activation and multiply
        var activated: f32;
        if (u.activation == 0u) {
            activated = silu(final_gate);
        } else {
            activated = gelu(final_gate);
        }

        output[col] = clamp_swiglu(activated * final_up * u.alpha);
    }
}

// Batched variant for prefill (M > 1)
// 2D dispatch: workgroup (x,y) computes 32 columns for batch row y
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

    // Which column within this workgroup (0..31)
    let col_in_wg = local_id / THREADS_PER_COL;
    // Which thread within the column (0..7)
    let tid_in_col = local_id % THREADS_PER_COL;

    // Global column index
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
                let gate_sm = get_scale_min_k4(gate_block.scales, sb);
                let gate_scale = gate_d * f32(gate_sm.x);
                let gate_min = gate_dmin * f32(gate_sm.y);

                let up_sm = get_scale_min_k4(up_block.scales, sb);
                let up_scale = up_d * f32(up_sm.x);
                let up_min = up_dmin * f32(up_sm.y);

                let sb_base = sb * SUBBLOCK_SIZE;

                for (var i: u32 = 0u; i < SUBBLOCK_SIZE; i = i + 4u) {
                    let k0 = k_base + sb_base + i;

                    let a0 = input[input_base + k0];
                    let a1 = input[input_base + k0 + 1u];
                    let a2 = input[input_base + k0 + 2u];
                    let a3 = input[input_base + k0 + 3u];

                    let gq0 = get_q4(gate_block.qs, sb_base + i);
                    let gq1 = get_q4(gate_block.qs, sb_base + i + 1u);
                    let gq2 = get_q4(gate_block.qs, sb_base + i + 2u);
                    let gq3 = get_q4(gate_block.qs, sb_base + i + 3u);

                    let gw0 = gate_scale * f32(gq0) - gate_min;
                    let gw1 = gate_scale * f32(gq1) - gate_min;
                    let gw2 = gate_scale * f32(gq2) - gate_min;
                    let gw3 = gate_scale * f32(gq3) - gate_min;

                    let uq0 = get_q4(up_block.qs, sb_base + i);
                    let uq1 = get_q4(up_block.qs, sb_base + i + 1u);
                    let uq2 = get_q4(up_block.qs, sb_base + i + 2u);
                    let uq3 = get_q4(up_block.qs, sb_base + i + 3u);

                    let uw0 = up_scale * f32(uq0) - up_min;
                    let uw1 = up_scale * f32(uq1) - up_min;
                    let uw2 = up_scale * f32(uq2) - up_min;
                    let uw3 = up_scale * f32(uq3) - up_min;

                    partial_gate = partial_gate + a0 * gw0 + a1 * gw1 + a2 * gw2 + a3 * gw3;
                    partial_up = partial_up + a0 * uw0 + a1 * uw1 + a2 * uw2 + a3 * uw3;
                }
            }
        }
    }

    // Reduction via shared memory
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
