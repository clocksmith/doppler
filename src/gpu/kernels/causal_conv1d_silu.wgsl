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
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn silu(x: f32) -> f32 {
    if (x >= 0.0) {
        let z = exp(-x);
        return x / (1.0 + z);
    }
    let z = exp(x);
    return x * z / (1.0 + z);
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
    var raw: f32 = 0.0;
    for (var kernel: u32 = 0u; kernel < u.kernel_size; kernel = kernel + 1u) {
        let source_token = i32(token) + i32(kernel) - i32(u.kernel_size) + 1;
        if (source_token < 0) {
            continue;
        }
        raw = raw
            + input[u32(source_token) * u.channels + channel]
            * weight[channel * u.kernel_size + kernel];
    }
    output[element] = silu(raw);
}
