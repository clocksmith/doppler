override WORKGROUP_SIZE: u32 = 128u;
const MAX_WORKGROUP_SIZE: u32 = 128u;

struct Uniforms {
    num_tokens: u32,
    num_heads: u32,
    key_dim: u32,
    value_dim: u32,
    query_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> query_key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> decay_beta: array<f32>;
@group(0) @binding(4) var<storage, read> state_history: array<f32>;
@group(0) @binding(5) var<storage, read> grad_output: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_query_key: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_value: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_decay_beta: array<f32>;
@group(0) @binding(9) var<storage, read_write> grad_state: array<f32>;

var<workgroup> shared_reduce: array<f32, MAX_WORKGROUP_SIZE>;

fn vector_index(token: u32, head: u32, dim: u32, width: u32) -> u32 {
    return ((token * u.num_heads + head) * width) + dim;
}

fn state_history_index(token: u32, head: u32, key_index: u32, value_index: u32) -> u32 {
    return ((((token * u.num_heads + head) * u.key_dim) + key_index) * u.value_dim) + value_index;
}

fn state_gradient_index(head: u32, key_index: u32, value_index: u32) -> u32 {
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
    let query_elements = u.num_tokens * u.num_heads * u.key_dim;
    let scalar_elements = u.num_tokens * u.num_heads;

    for (var reverse_token: u32 = 0u; reverse_token < u.num_tokens; reverse_token = reverse_token + 1u) {
        let token = u.num_tokens - 1u - reverse_token;
        let scalar_index = token * u.num_heads + head;
        let decay = exp(decay_beta[scalar_index]);
        var memory: f32 = 0.0;
        if (is_active) {
            for (var key_index: u32 = 0u; key_index < u.key_dim; key_index = key_index + 1u) {
                let previous_state = state_history[
                    state_history_index(token, head, key_index, lane)
                ];
                memory = memory
                    + previous_state
                    * decay
                    * query_key[query_elements + vector_index(token, head, key_index, u.key_dim)];
            }
        }
        let value_offset = vector_index(token, head, lane, u.value_dim);
        let value_at_token = select(0.0, value[value_offset], is_active);
        let delta = (value_at_token - memory) * decay_beta[scalar_elements + scalar_index];
        let output_gradient = select(0.0, grad_output[value_offset], is_active);

        for (var key_index: u32 = 0u; key_index < u.key_dim; key_index = key_index + 1u) {
            let query_offset = vector_index(token, head, key_index, u.key_dim);
            let next_state = select(
                0.0,
                state_history[state_history_index(token + 1u, head, key_index, lane)],
                is_active
            );
            shared_reduce[lane] = output_gradient * next_state * u.query_scale;
            workgroupBarrier();
            for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
                if (lane < stride) {
                    shared_reduce[lane] = shared_reduce[lane] + shared_reduce[lane + stride];
                }
                workgroupBarrier();
            }
            if (lane == 0u) {
                grad_query_key[query_offset] = shared_reduce[0];
            }
            if (is_active) {
                let gradient_state_offset = state_gradient_index(head, key_index, lane);
                grad_state[gradient_state_offset] = grad_state[gradient_state_offset]
                    + output_gradient
                    * query_key[query_offset]
                    * u.query_scale;
            }
            workgroupBarrier();
        }

        var grad_delta: f32 = 0.0;
        if (is_active) {
            for (var key_index: u32 = 0u; key_index < u.key_dim; key_index = key_index + 1u) {
                let gradient_state_offset = state_gradient_index(head, key_index, lane);
                grad_delta = grad_delta
                    + grad_state[gradient_state_offset]
                    * query_key[query_elements + vector_index(token, head, key_index, u.key_dim)];
            }
            grad_value[value_offset] = grad_delta * decay_beta[scalar_elements + scalar_index];
        }
        let grad_memory = -grad_delta * decay_beta[scalar_elements + scalar_index];
        shared_reduce[lane] = select(0.0, grad_delta * (value_at_token - memory), is_active);
        workgroupBarrier();
        for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
            if (lane < stride) {
                shared_reduce[lane] = shared_reduce[lane] + shared_reduce[lane + stride];
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            grad_decay_beta[scalar_elements + scalar_index] = shared_reduce[0];
        }
        workgroupBarrier();

        var local_grad_decay: f32 = 0.0;
        for (var key_index: u32 = 0u; key_index < u.key_dim; key_index = key_index + 1u) {
            let key_offset = vector_index(token, head, key_index, u.key_dim);
            let gradient_state_offset = state_gradient_index(head, key_index, lane);
            let previous_state = select(
                0.0,
                state_history[state_history_index(token, head, key_index, lane)],
                is_active
            );
            let decayed_state = previous_state * decay;
            let gradient_before_decay_update = select(0.0, grad_state[gradient_state_offset], is_active);

            shared_reduce[lane] = select(
                0.0,
                (gradient_before_decay_update * delta) + (grad_memory * decayed_state),
                is_active
            );
            workgroupBarrier();
            for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
                if (lane < stride) {
                    shared_reduce[lane] = shared_reduce[lane] + shared_reduce[lane + stride];
                }
                workgroupBarrier();
            }
            if (lane == 0u) {
                grad_query_key[query_elements + key_offset] = shared_reduce[0];
            }

            if (is_active) {
                let gradient_decayed_state = gradient_before_decay_update
                    + grad_memory * query_key[query_elements + key_offset];
                local_grad_decay = local_grad_decay + gradient_decayed_state * decayed_state;
                grad_state[gradient_state_offset] = gradient_decayed_state * decay;
            }
            workgroupBarrier();
        }

        shared_reduce[lane] = select(0.0, local_grad_decay, is_active);
        workgroupBarrier();
        for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
            if (lane < stride) {
                shared_reduce[lane] = shared_reduce[lane] + shared_reduce[lane + stride];
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            grad_decay_beta[scalar_index] = shared_reduce[0];
        }
        workgroupBarrier();
    }
}
