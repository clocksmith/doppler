// AUTO-GENERATED from src/gpu/kernels/attention_head256_f16kv.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Head-256 prefill, 16 subgroup lanes per query, compensated dot reduction.
// Fused Multi-Head Attention Kernel (head_dim 256, f16 KV)
//
// Fixed-shape prefill kernel for 256-dim heads. It reuses one shared 256-dim
// block for K and then V so the workgroup avoids the repeated 16-slice tiling
// used by the generic small-head path.

enable f16;
enable subgroups;

const BLOCK_SIZE: u32 = 32u;
const WORKGROUP_SIZE: u32 = 256u;
const HEAD_DIM: u32 = 256u;
const HEAD_DIM_VECS: u32 = 64u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    query_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,
    attn_softcap: f32,
    sliding_window: u32,
    kv_len_source: u32,
    kv_start: u32,
    page_size: u32,
    kv_layout: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;
@group(0) @binding(6) var<storage, read> page_table: array<u32>;

var<workgroup> shared_block: array<vec4<f16>, BLOCK_SIZE * HEAD_DIM_VECS>;

fn zero_vec4_f16() -> vec4<f16> {
    return vec4<f16>(f16(0.0), f16(0.0), f16(0.0), f16(0.0));
}

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn get_kv_pos(key_pos: u32) -> u32 {
    let abs_key = u.kv_start + key_pos;
    if (u.kv_layout == 1u && u.sliding_window > 0u) {
        return abs_key % u.sliding_window;
    }
    if (u.kv_layout == 2u) {
        let page_idx = abs_key / u.page_size;
        let in_page = abs_key - (page_idx * u.page_size);
        let phys_page = page_table[page_idx];
        return phys_page * u.page_size + in_page;
    }
    return abs_key;
}

fn get_kv_len() -> u32 {
    if (u.kv_len_source == 0u) {
        return u.seq_len;
    }
    return kv_len_buffer[0];
}

fn exp_refined(x: f32) -> f32 {
  if (abs(x) > 80.0) {
    return exp(x);
  }
  let n = round(x * 1.4426950408889634);
  var r = x;
  if (abs(x) >= 0.000000059604644775390625) {
    let bits = bitcast<u32>(abs(x));
    let mantissa = (bits & 0x007fffffu) | 0x00800000u;
    let shift = i32((bits >> 23u) & 255u) - 118;
    var input_fixed = 0u;
    if (shift >= 0) {
      input_fixed = mantissa << u32(shift);
    } else {
      let right = u32(-shift);
      input_fixed = (mantissa + (1u << (right - 1u))) >> right;
    }
    let power_fixed = u32(abs(n)) * 2977044472u;
    let residual = select(power_fixed - input_fixed, input_fixed - power_fixed, x >= 0.0);
    r = f32(bitcast<i32>(residual)) * 0.00000000023283064365386962890625;
  }
  var p = 0.0001984126984126984;
  p = fma(p, r, 0.001388888888888889);
  p = fma(p, r, 0.008333333333333333);
  p = fma(p, r, 0.041666666666666664);
  p = fma(p, r, 0.16666666666666666);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  return p * bitcast<f32>(u32(i32(n) + 127) << 23u);
}

fn reciprocal_refined(x: f32) -> f32 {
  let estimate = 1.0 / x;
  return fma(estimate, fma(-x, estimate, 1.0), estimate);
}


const QUERY_BLOCK: u32 = 16u;
const LANES_PER_QUERY: u32 = 16u;
var<workgroup> scores: array<f32, QUERY_BLOCK * BLOCK_SIZE>;
var<workgroup> factors: array<vec4<f32>, QUERY_BLOCK>;

fn dot_error(a: f32, b: f32, sum: f32) -> f32 {
  if (abs(a) >= abs(b)) { return fma(-1.0, sum, a) + b; }
  return fma(-1.0, sum, b) + a;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let thread = lid.x;
  let local_query = thread / LANES_PER_QUERY;
  let lane = thread % LANES_PER_QUERY;
  let head = wid.x % u.num_heads;
  let query = (wid.x / u.num_heads) * QUERY_BLOCK + local_query;
  let valid = query < u.query_len;
  let absolute_query = query + u.start_pos;
  let kv_head = get_kv_head_idx(head);
  let kv_length = get_kv_len();
  var q: array<vec4<f32>, 4>;
  var acc: array<vec4<f32>, 4>;
  for (var i = 0u; i < 4u; i++) {
    q[i] = vec4<f32>(0.0);
    acc[i] = vec4<f32>(0.0);
    if (valid) {
      let base = (query * u.num_heads + head) * HEAD_DIM + lane * 16u + i * 4u;
      q[i] = vec4<f32>(Q[base], Q[base + 1u], Q[base + 2u], Q[base + 3u]);
    }
  }
  var maximum = -3.402823e+38;
  var denominator = 0.0;
  for (var block = 0u; block * BLOCK_SIZE < kv_length; block++) {
    let key_start = block * BLOCK_SIZE;
    for (var index = thread; index < BLOCK_SIZE * HEAD_DIM_VECS; index += WORKGROUP_SIZE) {
      let key = key_start + index / HEAD_DIM_VECS;
      let dim4 = index % HEAD_DIM_VECS;
      var value = vec4<f16>(f16(0.0));
      if (key < kv_length) {
        let base = (get_kv_pos(key) * u.num_kv_heads + kv_head) * HEAD_DIM + dim4 * 4u;
        value = vec4<f16>(K[base], K[base + 1u], K[base + 2u], K[base + 3u]);
      }
      shared_block[index] = value;
    }
    workgroupBarrier();
    for (var key = 0u; key < BLOCK_SIZE; key++) {
      var sum = 0.0;
      var correction = 0.0;
      for (var i = 0u; i < 4u; i++) {
        let k = vec4<f32>(shared_block[key * HEAD_DIM_VECS + lane * 4u + i]);
        for (var component = 0u; component < 4u; component++) {
          let product = fma(q[i][component], k[component], 0.0);
          let next = fma(1.0, product, sum);
          correction += fma(q[i][component], k[component], -product) + dot_error(sum, product, next);
          sum = next;
        }
      }
      for (var offset = LANES_PER_QUERY / 2u; offset > 0u; offset /= 2u) {
        let other = subgroupShuffleDown(sum, offset);
        let other_correction = subgroupShuffleDown(correction, offset);
        let next = fma(1.0, other, sum);
        correction = correction + other_correction + dot_error(sum, other, next);
        sum = next;
      }
      if (lane == 0u) {
        let absolute_key = u.kv_start + key_start + key;
        var visible = valid && key_start + key < kv_length;
        if (u.is_causal != 0u && absolute_key > absolute_query) { visible = false; }
        if (u.sliding_window > 0u && absolute_query >= u.sliding_window
            && absolute_key < absolute_query - u.sliding_window + 1u) { visible = false; }
        var score = (sum + correction) * u.scale;
        if (u.attn_softcap > 0.0) { score = tanh(score / u.attn_softcap) * u.attn_softcap; }
        scores[local_query * BLOCK_SIZE + key] = select(-3.402823e+38, score, visible);
      }
    }
    workgroupBarrier();
    if (lane == 0u) {
      var new_maximum = maximum;
      for (var key = 0u; key < BLOCK_SIZE; key++) {
        new_maximum = max(new_maximum, scores[local_query * BLOCK_SIZE + key]);
      }
      var rescale = 0.0;
      if (denominator > 0.0) { rescale = exp_refined(maximum - new_maximum); }
      denominator = denominator * rescale;
      for (var key = 0u; key < BLOCK_SIZE; key++) {
        let index = local_query * BLOCK_SIZE + key;
        var probability = 0.0;
        if (scores[index] != -3.402823e+38) { probability = exp_refined(scores[index] - new_maximum); }
        scores[index] = probability;
        denominator += probability;
      }
      maximum = new_maximum;
      factors[local_query] = vec4<f32>(rescale, denominator, maximum, 0.0);
    }
    workgroupBarrier();
    for (var index = thread; index < BLOCK_SIZE * HEAD_DIM_VECS; index += WORKGROUP_SIZE) {
      let key = key_start + index / HEAD_DIM_VECS;
      let dim4 = index % HEAD_DIM_VECS;
      var value = vec4<f16>(f16(0.0));
      if (key < kv_length) {
        let base = (get_kv_pos(key) * u.num_kv_heads + kv_head) * HEAD_DIM + dim4 * 4u;
        value = vec4<f16>(V[base], V[base + 1u], V[base + 2u], V[base + 3u]);
      }
      shared_block[index] = value;
    }
    workgroupBarrier();
    for (var i = 0u; i < 4u; i++) { acc[i] *= factors[local_query].x; }
    if (valid) {
      for (var key = 0u; key < BLOCK_SIZE; key++) {
        let probability = scores[local_query * BLOCK_SIZE + key];
        if (probability == 0.0) { continue; }
        for (var i = 0u; i < 4u; i++) {
          acc[i] = fma(vec4<f32>(probability),
            vec4<f32>(shared_block[key * HEAD_DIM_VECS + lane * 4u + i]), acc[i]);
        }
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    var inv = 0.0;
    if (factors[local_query].y > 0.0) { inv = reciprocal_refined(factors[local_query].y); }
    for (var i = 0u; i < 4u; i++) {
      let base = (query * u.num_heads + head) * HEAD_DIM + lane * 16u + i * 4u;
      let value = acc[i] * inv;
      output[base] = value.x;
      output[base + 1u] = value.y;
      output[base + 2u] = value.z;
      output[base + 3u] = value.w;
    }
  }
}
