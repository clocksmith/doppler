// qwen4_exp QSA indexer preparation (Qwen4ExpTextQSAIndexer).
// Reads index_qk_proj output rows [QUERY_HEADS + 1, HEAD_DIM] per token.
// Query heads: RMSNorm over HEAD_DIM with (1 + q_layernorm weight), then
// rotate-half RoPE on the first ROTARY_DIM dims at the token's position.
// The key head is appended raw (pre-norm, pre-RoPE) to the owned key cache at
// cache row start_pos + token; every call appends, dense or sparse.

override WORKGROUP_SIZE: u32 = 128u;
override QUERY_HEADS: u32;
override HEAD_DIM: u32;
override ROTARY_DIM: u32;

const MAX_HEAD_DIM: u32 = 512u;

struct Uniforms {
    num_tokens: u32,
    start_pos: u32,     // cache row and position of the first token
    group_stride: u32,  // workgroups dispatched along x
    eps: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> projection: array<f32>;
@group(0) @binding(2) var<storage, read> q_norm_weight: array<f32>;
@group(0) @binding(3) var<storage, read> freqs_cos: array<f32>;   // [positions, ROTARY_DIM / 2]
@group(0) @binding(4) var<storage, read> freqs_sin: array<f32>;
@group(0) @binding(5) var<storage, read_write> query: array<f32>; // [num_tokens, QUERY_HEADS, HEAD_DIM]
@group(0) @binding(6) var<storage, read_write> raw_keys: array<f32>; // [capacity, HEAD_DIM]

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
    let group = wid.y * u.group_stride + wid.x;
    let heads_per_row = QUERY_HEADS + 1u;
    if (group >= u.num_tokens * heads_per_row || HEAD_DIM > MAX_HEAD_DIM) {
        return;
    }
    let token = group / heads_per_row;
    let head = group % heads_per_row;
    let local = lid.x;
    let source = token * heads_per_row * HEAD_DIM + head * HEAD_DIM;

    if (head == QUERY_HEADS) {
        let destination = (u.start_pos + token) * HEAD_DIM;
        for (var d = local; d < HEAD_DIM; d = d + WORKGROUP_SIZE) {
            raw_keys[destination + d] = projection[source + d];
        }
        return;
    }

    var sum = 0.0;
    for (var d = local; d < HEAD_DIM; d = d + WORKGROUP_SIZE) {
        let value = projection[source + d];
        sum = fma(value, value, sum);
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
        normalized[d] = (projection[source + d] * inverse_rms) * (1.0 + q_norm_weight[d]);
    }
    workgroupBarrier();

    let half = ROTARY_DIM / 2u;
    let table = (u.start_pos + token) * half;
    let destination = (token * QUERY_HEADS + head) * HEAD_DIM;
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
        query[destination + d] = value;
    }
}
