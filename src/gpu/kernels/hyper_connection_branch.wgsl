// Branch input of the qwen4_exp gated residual: the mean over streams of the
// sigmoid mix weights times the normalized streams. Each rounded product is
// accumulated in stream order, then the sum is divided by STREAM_COUNT.

override WORKGROUP_SIZE: u32 = 256u;
override STREAM_COUNT: u32;
override STREAM_WIDTH: u32;

struct Uniforms {
    size: u32,             // numTokens * STREAM_WIDTH
    dispatch_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> mix: array<f32>;
@group(0) @binding(2) var<storage, read> normed: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Pass each product to the adjacent invocation through workgroup memory.
// A local multiply helper alone can be contracted into the subsequent add.
// The cross-invocation exchange makes the source's F32 product materialization
// observable before accumulation, without a full-sized intermediate buffer.
var<workgroup> rounded_products: array<f32, WORKGROUP_SIZE>;

fn divide_refined(x: f32, y: f32) -> f32 {
  let estimate = 1.0 / y;
  let inverse = fma(estimate, fma(-y, estimate, 1.0), estimate);
  let quotient = x * inverse;
  return fma(fma(-quotient, y, x), inverse, quotient);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let index = gid.y * u.dispatch_stride + gid.x;
    let group_base = index - lid.x;
    let product_index = group_base + (lid.x + 1u) % WORKGROUP_SIZE;
    let token = product_index / STREAM_WIDTH;
    let column = product_index % STREAM_WIDTH;
    let token_base = token * STREAM_COUNT * STREAM_WIDTH + column;
    var sum = 0.0;
    for (var stream = 0u; stream < STREAM_COUNT; stream = stream + 1u) {
        var product = 0.0;
        if (product_index < u.size) {
            let offset = token_base + stream * STREAM_WIDTH;
            product = mix[offset] * normed[offset];
        }
        rounded_products[lid.x] = product;
        workgroupBarrier();
        sum = sum + rounded_products[(lid.x + WORKGROUP_SIZE - 1u) % WORKGROUP_SIZE];
        workgroupBarrier();
    }
    if (index < u.size) {
        output[index] = divide_refined(sum, f32(STREAM_COUNT));
    }
}
