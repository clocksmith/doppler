override WORKGROUP_SIZE: u32 = 128u;

struct Uniforms {
    num_tokens: u32,
    total_tokens: u32,
    token_offset: u32,
    num_heads: u32,
    key_dim: u32,
    value_dim: u32,
    checkpoint_interval: u32,
    checkpoint_count: u32,
    query_scale: f32,
    initial_state_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> query_key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> decay_beta: array<f32>;
@group(0) @binding(4) var<storage, read> initial_state: array<f32>;
@group(0) @binding(5) var<storage, read_write> current_state: array<f32>;
@group(0) @binding(6) var<storage, read_write> checkpoints: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;

fn vector_index(token: u32, head: u32, dim: u32, width: u32) -> u32 {
    return ((token * u.num_heads + head) * width) + dim;
}

fn state_index(head: u32, key_index: u32, value_index: u32) -> u32 {
    return (((head * u.key_dim) + key_index) * u.value_dim) + value_index;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let head = wid.x;
    let lane = lid.x;
    if (head >= u.num_heads) {
        return;
    }
    let is_active = lane < u.value_dim;
    let query_elements = u.total_tokens * u.num_heads * u.key_dim;
    let scalar_elements = u.total_tokens * u.num_heads;
    let state_elements = u.num_heads * u.key_dim * u.value_dim;

    if (is_active) {
        for (var key_index: u32 = 0u; key_index < u.key_dim; key_index = key_index + 1u) {
            let offset = state_index(head, key_index, lane);
            let state_value = initial_state[u.initial_state_offset + offset];
            current_state[offset] = state_value;
            checkpoints[offset] = state_value;
        }
    }
    workgroupBarrier();

    for (var local_token: u32 = 0u; local_token < u.num_tokens; local_token = local_token + 1u) {
        let token = u.token_offset + local_token;
        let scalar_index = token * u.num_heads + head;
        let decay = exp(decay_beta[scalar_index]);
        var memory: f32 = 0.0;
        if (is_active) {
            for (var key_index: u32 = 0u; key_index < u.key_dim; key_index = key_index + 1u) {
                let offset = state_index(head, key_index, lane);
                current_state[offset] = current_state[offset] * decay;
                memory = memory
                    + current_state[offset]
                    * query_key[query_elements + vector_index(token, head, key_index, u.key_dim)];
            }
        }
        let value_offset = vector_index(token, head, lane, u.value_dim);
        let value_at_token = select(0.0, value[value_offset], is_active);
        let delta = (value_at_token - memory) * decay_beta[scalar_elements + scalar_index];
        var output_value: f32 = 0.0;
        if (is_active) {
            for (var key_index: u32 = 0u; key_index < u.key_dim; key_index = key_index + 1u) {
                let offset = state_index(head, key_index, lane);
                let key_value = query_key[
                    query_elements + vector_index(token, head, key_index, u.key_dim)
                ];
                current_state[offset] = current_state[offset] + key_value * delta;
                output_value = output_value
                    + current_state[offset]
                    * query_key[vector_index(token, head, key_index, u.key_dim)]
                    * u.query_scale;
            }
            output[value_offset] = output_value;
        }

        let completed_tokens = local_token + 1u;
        let is_checkpoint = completed_tokens % u.checkpoint_interval == 0u
            || completed_tokens == u.num_tokens;
        if (is_active && is_checkpoint) {
            let checkpoint_index = min(
                (completed_tokens + u.checkpoint_interval - 1u) / u.checkpoint_interval,
                u.checkpoint_count
            );
            let checkpoint_base = checkpoint_index * state_elements;
            for (var key_index: u32 = 0u; key_index < u.key_dim; key_index = key_index + 1u) {
                let offset = state_index(head, key_index, lane);
                checkpoints[checkpoint_base + offset] = current_state[offset];
            }
        }
        workgroupBarrier();
    }
}
