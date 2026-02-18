// Fused FFN Kernel - Native F16 I/O
//
// Full f16 pipeline variant: f16 input, f16 weights, f16 output.
// f32 accumulators and shared memory for numerical stability.
// Requires 'shader-f16' feature.
//
// Standard FFN (SwiGLU/GeGLU):
//   gate = x @ W_gate
//   up = x @ W_up
//   hidden = activation(gate) * up
//   out = hidden @ W_down
//
// This kernel fuses steps 1-3 into a single dispatch.
// Input is widened f16→f32 on shared memory load; output is narrowed f32→f16 on write.

enable f16;
enable subgroups;

override WORKGROUP_SIZE: u32 = 256u;
const SHARED_INPUT_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 256u;

struct Uniforms {
    M: u32,
    hidden_size: u32,
    intermediate_size: u32,
    alpha: f32,
    activation: u32,
    clamp_max: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> W_gate: array<f16>;
@group(0) @binding(3) var<storage, read> W_up: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;

var<workgroup> shared_input: array<f32, SHARED_INPUT_SIZE>;
var<workgroup> sg_sums: array<f32, MAX_SUBGROUPS * 2u>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn gelu(x: f32) -> f32 {
    let c = 0.7978845608;
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

fn clamp_swiglu(x: f32) -> f32 {
    if (u.clamp_max <= 0.0 || u.activation != 0u) {
        return x;
    }
    return clamp(x, -u.clamp_max, u.clamp_max);
}

// Single-output-per-workgroup variant for decode (M=1)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let out_idx = wg_id.x;
    let tid = lid.x;
    let hidden_size = u.hidden_size;

    if (out_idx >= u.intermediate_size) {
        return;
    }

    let subgroup_id = tid / sg_size;
    let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;

    var gate_sum: f32 = 0.0;
    var up_sum: f32 = 0.0;

    let weight_base = out_idx * hidden_size;

    let num_tiles = (hidden_size + SHARED_INPUT_SIZE - 1u) / SHARED_INPUT_SIZE;
    for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
        let tile_start = tile * SHARED_INPUT_SIZE;
        let tile_end = min(tile_start + SHARED_INPUT_SIZE, hidden_size);

        // Load f16 input, widen to f32 in shared memory for accumulation precision
        for (var i = tid; i < SHARED_INPUT_SIZE; i = i + WORKGROUP_SIZE) {
            let idx = tile_start + i;
            if (idx < hidden_size) {
                shared_input[i] = f32(input[idx]);
            }
        }
        workgroupBarrier();

        for (var k = tile_start + tid; k < tile_end; k = k + WORKGROUP_SIZE) {
            let x = shared_input[k - tile_start];
            gate_sum += x * f32(W_gate[weight_base + k]);
            up_sum += x * f32(W_up[weight_base + k]);
        }

        workgroupBarrier();
    }

    let sg_gate = subgroupAdd(gate_sum);
    let sg_up = subgroupAdd(up_sum);

    if (sg_id == 0u && subgroup_id < num_subgroups) {
        sg_sums[subgroup_id] = sg_gate;
        sg_sums[subgroup_id + MAX_SUBGROUPS] = sg_up;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var final_gate: f32 = 0.0;
        var final_up: f32 = 0.0;
        for (var s = 0u; s < num_subgroups; s++) {
            final_gate += sg_sums[s];
            final_up += sg_sums[s + MAX_SUBGROUPS];
        }

        var activated: f32;
        if (u.activation == 0u) {
            activated = silu(final_gate);
        } else {
            activated = gelu(final_gate);
        }

        output[out_idx] = f16(clamp_swiglu(activated * final_up * u.alpha));
    }
}

// Batched variant for prefill (M > 1)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_batched(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let out_idx = wg_id.x;
    let batch_idx = wg_id.y;
    let tid = lid.x;
    let hidden_size = u.hidden_size;
    let intermediate_size = u.intermediate_size;

    if (out_idx >= intermediate_size || batch_idx >= u.M) {
        return;
    }

    let subgroup_id = tid / sg_size;
    let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;

    var gate_sum: f32 = 0.0;
    var up_sum: f32 = 0.0;

    let weight_base = out_idx * hidden_size;
    let input_base = batch_idx * hidden_size;

    let num_tiles = (hidden_size + SHARED_INPUT_SIZE - 1u) / SHARED_INPUT_SIZE;
    for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
        let tile_start = tile * SHARED_INPUT_SIZE;
        let tile_end = min(tile_start + SHARED_INPUT_SIZE, hidden_size);

        // Load f16 input, widen to f32 in shared memory for accumulation precision
        for (var i = tid; i < SHARED_INPUT_SIZE; i = i + WORKGROUP_SIZE) {
            let idx = tile_start + i;
            if (idx < hidden_size) {
                shared_input[i] = f32(input[input_base + idx]);
            }
        }
        workgroupBarrier();

        for (var k = tile_start + tid; k < tile_end; k = k + WORKGROUP_SIZE) {
            let x = shared_input[k - tile_start];
            gate_sum += x * f32(W_gate[weight_base + k]);
            up_sum += x * f32(W_up[weight_base + k]);
        }

        workgroupBarrier();
    }

    let sg_gate = subgroupAdd(gate_sum);
    let sg_up = subgroupAdd(up_sum);

    if (sg_id == 0u && subgroup_id < num_subgroups) {
        sg_sums[subgroup_id] = sg_gate;
        sg_sums[subgroup_id + MAX_SUBGROUPS] = sg_up;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var final_gate: f32 = 0.0;
        var final_up: f32 = 0.0;
        for (var s = 0u; s < num_subgroups; s++) {
            final_gate += sg_sums[s];
            final_up += sg_sums[s + MAX_SUBGROUPS];
        }

        var activated: f32;
        if (u.activation == 0u) {
            activated = silu(final_gate);
        } else {
            activated = gelu(final_gate);
        }

        let out_offset = batch_idx * intermediate_size + out_idx;
        output[out_offset] = f16(clamp_swiglu(activated * final_up * u.alpha));
    }
}
