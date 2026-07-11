// Fused FFN batched prefill for packed f16 weights and f32 activations.
//
// This variant keeps the AF32 lane intact while fusing gate_proj + up_proj +
// activation for f16-materialized MLP weights.

enable subgroups;

override WORKGROUP_SIZE: u32 = 256u;
override SHARED_INPUT_SIZE: u32 = 256u;
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
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> W_gate: array<f32>;
@group(0) @binding(3) var<storage, read> W_up: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> shared_input: array<f32, SHARED_INPUT_SIZE>;
var<workgroup> sg_sums: array<f32, MAX_SUBGROUPS * 2u>;

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

    let gate_base = out_idx * hidden_size;
    let up_base = out_idx * hidden_size;
    let input_base = batch_idx * hidden_size;

    let num_tiles = (hidden_size + SHARED_INPUT_SIZE - 1u) / SHARED_INPUT_SIZE;
    for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
        let tile_start = tile * SHARED_INPUT_SIZE;
        let tile_end = min(tile_start + SHARED_INPUT_SIZE, hidden_size);

        for (var i = tid; i < SHARED_INPUT_SIZE; i = i + WORKGROUP_SIZE) {
            let idx = tile_start + i;
            if (idx < hidden_size) {
                shared_input[i] = input[input_base + idx];
            }
        }
        workgroupBarrier();

        for (var k = tile_start + tid; k < tile_end; k = k + WORKGROUP_SIZE) {
            let x = shared_input[k - tile_start];

            let gate_packed = W_gate[gate_base / 2u + k / 2u];
            let up_packed = W_up[up_base / 2u + k / 2u];

            let gate_vec = unpack2x16float(bitcast<u32>(gate_packed));
            let up_vec = unpack2x16float(bitcast<u32>(up_packed));

            let g = select(gate_vec.y, gate_vec.x, (k % 2u) == 0u);
            let v = select(up_vec.y, up_vec.x, (k % 2u) == 0u);

            gate_sum += x * g;
            up_sum += x * v;
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
        output[out_offset] = clamp_swiglu(activated * final_up * u.alpha);
    }
}
