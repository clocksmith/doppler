// qwen4_exp QSA block keys for newly completed blocks.
// A block holds COMPRESS_RATIO consecutive cached tokens starting at
// block * COMPRESS_RATIO. Its key is RMSNorm(mean of the raw keys) with
// (1 + k_layernorm weight), rotated (rotate-half, first ROTARY_DIM dims) at
// the position of the block's first token. Positions equal cache rows: the
// owning cache admits only contiguous text positions.

override WORKGROUP_SIZE: u32 = 128u;
override HEAD_DIM: u32;
override ROTARY_DIM: u32;
override COMPRESS_RATIO: u32;

const MAX_HEAD_DIM: u32 = 512u;

struct Uniforms {
    first_block: u32,
    block_count: u32,
    group_stride: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> raw_keys: array<f32>;      // [capacity, HEAD_DIM]
@group(0) @binding(2) var<storage, read> k_norm_weight: array<f32>;
@group(0) @binding(3) var<storage, read> freqs_cos: array<f32>;     // [positions, ROTARY_DIM / 2]
@group(0) @binding(4) var<storage, read> freqs_sin: array<f32>;
@group(0) @binding(5) var<storage, read_write> block_keys: array<f32>; // [capacity / COMPRESS_RATIO, HEAD_DIM]

var<workgroup> partial: array<f32, WORKGROUP_SIZE>;
var<workgroup> normalized: array<f32, MAX_HEAD_DIM>;

fn inverse_root_refined(x: f32) -> f32 {
  let estimate = inverseSqrt(x);
  let squared = estimate * estimate;
  let error = fma(estimate, estimate, -squared);
  let residual = fma(-x, squared, 1.0) - x * error;
  return fma(0.5 * estimate, residual, estimate);
}

fn divide_refined(x: f32, y: f32) -> f32 {
  let estimate = 1.0 / y;
  let inverse = fma(estimate, fma(-y, estimate, 1.0), estimate);
  let quotient = x * inverse;
  return fma(fma(-quotient, y, x), inverse, quotient);
}

fn multiply_ordered(a: f32, b: f32) -> f32 {
  var product = 1.0;
  let factors = array<f32, 2>(a, b);
  for (var i = 0u; i < 2u; i = i + 1u) {
    product = fma(product, factors[i], 0.0);
  }
  return product;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let index = wid.y * u.group_stride + wid.x;
    if (index >= u.block_count || HEAD_DIM > MAX_HEAD_DIM) {
        return;
    }
    let block = u.first_block + index;
    let first_token = block * COMPRESS_RATIO;
    let local = lid.x;

    var sum = 0.0;
    for (var d = local; d < HEAD_DIM; d = d + WORKGROUP_SIZE) {
        var pooled = 0.0;
        for (var r = 0u; r < COMPRESS_RATIO; r = r + 1u) {
            pooled = pooled + raw_keys[(first_token + r) * HEAD_DIM + d];
        }
        pooled = divide_refined(pooled, f32(COMPRESS_RATIO));
        normalized[d] = pooled;
        sum = fma(pooled, pooled, sum);
    }
    partial[local] = sum;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) {
            partial[local] = partial[local] + partial[local + stride];
        }
        workgroupBarrier();
    }
    let inverse_rms = inverse_root_refined(divide_refined(partial[0], f32(HEAD_DIM)) + u.eps);
    for (var d = local; d < HEAD_DIM; d = d + WORKGROUP_SIZE) {
        normalized[d] = (normalized[d] * inverse_rms) * (1.0 + k_norm_weight[d]);
    }
    workgroupBarrier();

    let half = ROTARY_DIM / 2u;
    let table = first_token * half;
    let destination = block * HEAD_DIM;
    for (var d = local; d < HEAD_DIM; d = d + WORKGROUP_SIZE) {
        var value = normalized[d];
        if (d < ROTARY_DIM) {
            let pair = d % half;
            let c = freqs_cos[table + pair];
            let s = freqs_sin[table + pair];
            if (d < half) {
                value = multiply_ordered(normalized[d], c) - multiply_ordered(normalized[d + half], s);
            } else {
                value = multiply_ordered(normalized[d], c) + multiply_ordered(normalized[d - half], s);
            }
        }
        block_keys[destination + d] = value;
    }
}
