override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    channels: u32,
    kernel_size: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;

fn sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        let z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    let z = exp(x);
    return z / (1.0 + z);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let element = gid.x;
    let element_count = u.num_tokens * u.channels;
    if (element >= element_count) {
        return;
    }

    let token = element / u.channels;
    let channel = element % u.channels;
    let last_output_token = min(u.num_tokens - 1u, token + u.kernel_size - 1u);
    var gradient: f32 = 0.0;

    for (var output_token = token; output_token <= last_output_token; output_token = output_token + 1u) {
        let kernel_index = token + u.kernel_size - 1u - output_token;
        var raw: f32 = 0.0;
        for (var kernel: u32 = 0u; kernel < u.kernel_size; kernel = kernel + 1u) {
            let padded_source = i32(output_token) + i32(kernel) - i32(u.kernel_size) + 1;
            if (padded_source < 0) {
                continue;
            }
            let source = u32(padded_source) * u.channels + channel;
            raw = raw + input[source] * weight[channel * u.kernel_size + kernel];
        }
        let probability = sigmoid(raw);
        let silu_derivative = probability * (1.0 + raw * (1.0 - probability));
        let output_offset = output_token * u.channels + channel;
        gradient = gradient
            + grad_output[output_offset]
            * silu_derivative
            * weight[channel * u.kernel_size + kernel_index];
    }

    grad_input[element] = gradient;
}
