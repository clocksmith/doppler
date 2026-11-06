// Injection of a branch output into the qwen4_exp residual streams:
// streams[t, s, w] + branch[t, w] * gate[t, s], with the product rounded
// before the addition (Qwen4ExpTextDecoderLayer injection order). The branch
// output never passes through another normalization or residual operation.

override WORKGROUP_SIZE: u32 = 256u;
override STREAM_COUNT: u32;
override STREAM_WIDTH: u32;

struct Uniforms {
    size: u32,             // numTokens * STREAM_COUNT * STREAM_WIDTH
    dispatch_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> streams: array<f32>;
@group(0) @binding(2) var<storage, read> branch: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Cross-invocation storage prevents contraction with the residual add. Local
// multiply helpers do not establish the source's F32 materialization boundary.
var<workgroup> rounded_products: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let index = gid.y * u.dispatch_stride + gid.x;
    let group_base = index - lid.x;
    let product_index = group_base + (lid.x + 1u) % WORKGROUP_SIZE;
    let token_width = STREAM_COUNT * STREAM_WIDTH;
    let token = product_index / token_width;
    let within = product_index % token_width;
    let stream = within / STREAM_WIDTH;
    let column = within % STREAM_WIDTH;
    var product = 0.0;
    if (product_index < u.size) {
        product = branch[token * STREAM_WIDTH + column] * gate[token * STREAM_COUNT + stream];
    }
    rounded_products[lid.x] = product;
    workgroupBarrier();
    if (index < u.size) {
        output[index] = streams[index] + rounded_products[(lid.x + WORKGROUP_SIZE - 1u) % WORKGROUP_SIZE];
    }
}
