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

fn multiply_ordered(a: f32, b: f32) -> f32 {
  var product = 1.0;
  let factors = array<f32, 2>(a, b);
  for (var i = 0u; i < 2u; i = i + 1u) {
    product = fma(product, factors[i], 0.0);
  }
  return product;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.y * u.dispatch_stride + gid.x;
    if (index >= u.size) { return; }
    let token_width = STREAM_COUNT * STREAM_WIDTH;
    let token = index / token_width;
    let within = index % token_width;
    let stream = within / STREAM_WIDTH;
    let column = within % STREAM_WIDTH;
    let injection = multiply_ordered(branch[token * STREAM_WIDTH + column], gate[token * STREAM_COUNT + stream]);
    output[index] = streams[index] + injection;
}
