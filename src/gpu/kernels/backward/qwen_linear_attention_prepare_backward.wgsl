override WORKGROUP_SIZE: u32 = 128u;
const MAX_WORKGROUP_SIZE: u32 = 128u;

struct Uniforms {
    num_tokens: u32,
    num_key_heads: u32,
    num_value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    conv_size: u32,
    query_size: u32,
    key_size: u32,
    value_size: u32,
    repeat_factor: u32,
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> mixed: array<f32>;
@group(0) @binding(2) var<storage, read> packed_a_b: array<f32>;
@group(0) @binding(3) var<storage, read> packed_parameters: array<f32>;
@group(0) @binding(4) var<storage, read> packed_grad_query_key: array<f32>;
@group(0) @binding(5) var<storage, read> grad_value: array<f32>;
@group(0) @binding(6) var<storage, read> packed_grad_decay_beta: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_mixed: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_a: array<f32>;
@group(0) @binding(9) var<storage, read_write> grad_b: array<f32>;

var<workgroup> shared_sum_squares: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_dot: array<f32, MAX_WORKGROUP_SIZE>;

fn stable_sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        let z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    let z = exp(x);
    return z / (1.0 + z);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let workgroup = wid.x;
    let lane = lid.x;
    let source_rows = u.num_tokens * u.num_key_heads;
    let query_key_rows = source_rows * 2u;
    let repeated_elements = u.num_tokens * u.num_value_heads * u.key_dim;
    if (workgroup < query_key_rows) {
        let is_key = workgroup >= source_rows;
        let source_row = workgroup % source_rows;
        let token = source_row / u.num_key_heads;
        let source_head = source_row % u.num_key_heads;
        let section_offset = select(0u, u.query_size, is_key);
        let source_base = token * u.conv_size + section_offset + source_head * u.key_dim;
        let gradient_section = select(0u, repeated_elements, is_key);
        var sum_squares = 0.0;
        var dot = 0.0;
        for (var dim = lane; dim < u.key_dim; dim = dim + WORKGROUP_SIZE) {
            var aggregated_gradient = 0.0;
            for (var repeat = 0u; repeat < u.repeat_factor; repeat = repeat + 1u) {
                let destination_head = source_head * u.repeat_factor + repeat;
                let destination = ((token * u.num_value_heads + destination_head) * u.key_dim) + dim;
                aggregated_gradient = aggregated_gradient
                    + packed_grad_query_key[gradient_section + destination];
            }
            let item = mixed[source_base + dim];
            sum_squares = sum_squares + item * item;
            dot = dot + aggregated_gradient * item;
        }
        shared_sum_squares[lane] = sum_squares;
        shared_dot[lane] = dot;
        workgroupBarrier();
        var stride = WORKGROUP_SIZE / 2u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (lane < stride) {
                shared_sum_squares[lane] = shared_sum_squares[lane]
                    + shared_sum_squares[lane + stride];
                shared_dot[lane] = shared_dot[lane] + shared_dot[lane + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let inverse_norm = inverseSqrt(shared_sum_squares[0] + u.eps);
        let correction = shared_dot[0] * inverse_norm * inverse_norm;
        for (var dim = lane; dim < u.key_dim; dim = dim + WORKGROUP_SIZE) {
            var aggregated_gradient = 0.0;
            for (var repeat = 0u; repeat < u.repeat_factor; repeat = repeat + 1u) {
                let destination_head = source_head * u.repeat_factor + repeat;
                let destination = ((token * u.num_value_heads + destination_head) * u.key_dim) + dim;
                aggregated_gradient = aggregated_gradient
                    + packed_grad_query_key[gradient_section + destination];
            }
            grad_mixed[source_base + dim] = inverse_norm
                * (aggregated_gradient - mixed[source_base + dim] * correction);
        }
        return;
    }

    let linear_index = (workgroup - query_key_rows) * WORKGROUP_SIZE + lane;
    let value_elements = u.num_tokens * u.value_size;
    let scalar_elements = u.num_tokens * u.num_value_heads;
    if (linear_index < value_elements) {
        let token = linear_index / u.value_size;
        let column = linear_index % u.value_size;
        grad_mixed[token * u.conv_size + u.query_size + u.key_size + column]
            = grad_value[linear_index];
        return;
    }
    let scalar_linear = linear_index - value_elements;
    if (scalar_linear < scalar_elements) {
        let head = scalar_linear % u.num_value_heads;
        let a_log = packed_parameters[head];
        let dt_bias = packed_parameters[u.num_value_heads + head];
        let a_value = packed_a_b[scalar_linear];
        let softplus_derivative = stable_sigmoid(a_value + dt_bias);
        grad_a[scalar_linear] = packed_grad_decay_beta[scalar_linear]
            * -exp(a_log)
            * softplus_derivative;
        return;
    }
    let beta_index = scalar_linear - scalar_elements;
    if (beta_index < scalar_elements) {
        let b_value = packed_a_b[scalar_elements + beta_index];
        let beta_value = stable_sigmoid(b_value);
        grad_b[beta_index] = packed_grad_decay_beta[scalar_elements + beta_index]
            * beta_value
            * (1.0 - beta_value);
    }
}
