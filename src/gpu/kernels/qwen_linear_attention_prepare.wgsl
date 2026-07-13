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
@group(0) @binding(2) var<storage, read> a_projection: array<f32>;
@group(0) @binding(3) var<storage, read> b_projection: array<f32>;
@group(0) @binding(4) var<storage, read> packed_parameters: array<f32>;
@group(0) @binding(5) var<storage, read_write> query: array<f32>;
@group(0) @binding(6) var<storage, read_write> key: array<f32>;
@group(0) @binding(7) var<storage, read_write> value: array<f32>;
@group(0) @binding(8) var<storage, read_write> log_decay: array<f32>;
@group(0) @binding(9) var<storage, read_write> beta: array<f32>;

var<workgroup> shared_sum_squares: array<f32, MAX_WORKGROUP_SIZE>;

fn stable_sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        let z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    let z = exp(x);
    return z / (1.0 + z);
}

fn stable_softplus(x: f32) -> f32 {
    return log(1.0 + exp(-abs(x))) + max(x, 0.0);
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
    if (workgroup < query_key_rows) {
        let is_key = workgroup >= source_rows;
        let source_row = workgroup % source_rows;
        let token = source_row / u.num_key_heads;
        let source_head = source_row % u.num_key_heads;
        let section_offset = select(0u, u.query_size, is_key);
        let source_base = token * u.conv_size + section_offset + source_head * u.key_dim;
        var sum_squares = 0.0;
        for (var dim = lane; dim < u.key_dim; dim = dim + WORKGROUP_SIZE) {
            let item = mixed[source_base + dim];
            sum_squares = sum_squares + item * item;
        }
        shared_sum_squares[lane] = sum_squares;
        workgroupBarrier();
        var stride = WORKGROUP_SIZE / 2u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (lane < stride) {
                shared_sum_squares[lane] = shared_sum_squares[lane]
                    + shared_sum_squares[lane + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let inverse_norm = inverseSqrt(shared_sum_squares[0] + u.eps);
        for (var dim = lane; dim < u.key_dim; dim = dim + WORKGROUP_SIZE) {
            let normalized = mixed[source_base + dim] * inverse_norm;
            for (var repeat = 0u; repeat < u.repeat_factor; repeat = repeat + 1u) {
                let destination_head = source_head * u.repeat_factor + repeat;
                let destination = ((token * u.num_value_heads + destination_head) * u.key_dim) + dim;
                if (is_key) {
                    key[destination] = normalized;
                } else {
                    query[destination] = normalized;
                }
            }
        }
        return;
    }

    let linear_index = (workgroup - query_key_rows) * WORKGROUP_SIZE + lane;
    let value_elements = u.num_tokens * u.value_size;
    let scalar_elements = u.num_tokens * u.num_value_heads;
    if (linear_index < value_elements) {
        let token = linear_index / u.value_size;
        let column = linear_index % u.value_size;
        value[linear_index] = mixed[
            token * u.conv_size + u.query_size + u.key_size + column
        ];
        return;
    }
    let scalar_linear = linear_index - value_elements;
    if (scalar_linear < scalar_elements) {
        let head = scalar_linear % u.num_value_heads;
        let a_log = packed_parameters[head];
        let dt_bias = packed_parameters[u.num_value_heads + head];
        log_decay[scalar_linear] = -exp(a_log)
            * stable_softplus(a_projection[scalar_linear] + dt_bias);
        return;
    }
    let beta_index = scalar_linear - scalar_elements;
    if (beta_index < scalar_elements) {
        beta[beta_index] = stable_sigmoid(b_projection[beta_index]);
    }
}
