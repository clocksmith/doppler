// lm_head_select_logits.wgsl

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

const MAX_WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    hidden_size: u32,
    vocab_size: u32,
    token_count: u32,
    hidden_offset: u32,
    transpose_b: u32,
    logit_softcap: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> hidden: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f16>;
@group(0) @binding(3) var<storage, read> token_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> partial_sums: array<f32, MAX_WORKGROUP_SIZE>;

fn apply_softcap(x: f32, softcap: f32) -> f32 {
    if (softcap <= 0.0) {
        return x;
    }
    return softcap * tanh(x / softcap);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let local_id = lid.x;
    let token_index = wg_id.x;
    if (token_index >= u.token_count) {
        return;
    }

    let token_id = token_ids[token_index];
    var sum: f32 = 0.0;
    var k = local_id;
    if (u.transpose_b == 1u) {
        let row_offset = token_id * u.hidden_size;
        while (k < u.hidden_size) {
            sum = sum + hidden[u.hidden_offset + k] * f32(weights[row_offset + k]);
            k = k + WORKGROUP_SIZE;
        }
    } else {
        while (k < u.hidden_size) {
            sum = sum + hidden[u.hidden_offset + k] * f32(weights[k * u.vocab_size + token_id]);
            k = k + WORKGROUP_SIZE;
        }
    }

    partial_sums[local_id] = sum;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            partial_sums[local_id] = partial_sums[local_id] + partial_sums[local_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_id == 0u) {
        output[token_index] = apply_softcap(partial_sums[0], u.logit_softcap);
    }
}
