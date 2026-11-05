// Attention over QSA-selected tokens (qwen4_exp indexed attention).
// One workgroup per (query token, head). Query rows are f32 after Q norm and
// RoPE; keys and values are the original per-token f16 cache rows (contiguous
// [position, kv_head, head_dim] layout). selection_meta[token] = (count, mode):
// mode 0 attends to cache rows [0, count); mode 1 attends to the listed rows.
// Softmax is computed online in f32 over chunks; excluded tokens contribute
// nothing, matching the source's additive mask with exactly zero probability.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

const MAX_HEAD_DIM: u32 = 512u;
const NEG_LARGE: f32 = -3.402823e+38;

struct Uniforms {
    num_tokens: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_selected: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;                 // [num_tokens, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> K: array<f16>;                 // [positions, num_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read> selection: array<u32>;         // [num_tokens, max_selected]
@group(0) @binding(5) var<storage, read> selection_meta: array<u32>;    // [num_tokens, 2]
@group(0) @binding(6) var<storage, read_write> output: array<f32>;      // [num_tokens, num_heads, head_dim]

var<workgroup> shared_q: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_p: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_token: array<u32, WORKGROUP_SIZE>;
var<workgroup> reduce: array<f32, WORKGROUP_SIZE>;

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

fn tree_max(local: u32, value: f32) -> f32 {
    reduce[local] = value;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) {
            reduce[local] = max(reduce[local], reduce[local + stride]);
        }
        workgroupBarrier();
    }
    let result = reduce[0];
    workgroupBarrier();
    return result;
}

fn tree_sum(local: u32, value: f32) -> f32 {
    reduce[local] = value;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) {
            reduce[local] = reduce[local] + reduce[local + stride];
        }
        workgroupBarrier();
    }
    let result = reduce[0];
    workgroupBarrier();
    return result;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let head = wid.x;
    let token = wid.y;
    let head_dim = u.head_dim;
    if (head >= u.num_heads || token >= u.num_tokens || head_dim > MAX_HEAD_DIM || head_dim > 2u * WORKGROUP_SIZE) {
        return;
    }
    let local = lid.x;
    let count = selection_meta[token * 2u];
    let listed = selection_meta[token * 2u + 1u] == 1u;
    let kv_head = head / (u.num_heads / u.num_kv_heads);
    let row_stride = u.num_kv_heads * head_dim;
    let q_base = (token * u.num_heads + head) * head_dim;
    for (var d = local; d < head_dim; d = d + WORKGROUP_SIZE) {
        shared_q[d] = Q[q_base + d];
    }
    workgroupBarrier();

    let dim0 = local;
    let dim1 = local + WORKGROUP_SIZE;
    var acc0 = 0.0;
    var acc1 = 0.0;
    var running_max = NEG_LARGE;
    var running_sum = 0.0;

    let chunks = (count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    for (var chunk = 0u; chunk < chunks; chunk = chunk + 1u) {
        let index = chunk * WORKGROUP_SIZE + local;
        let valid = index < count;
        var cached = 0u;
        var score = NEG_LARGE;
        if (valid) {
            cached = select(index, selection[token * u.max_selected + index], listed);
            let k_base = cached * row_stride + kv_head * head_dim;
            var dot = 0.0;
            for (var d = 0u; d < head_dim; d = d + 1u) {
                dot = fma(shared_q[d], f32(K[k_base + d]), dot);
            }
            score = dot * u.scale;
        }
        shared_token[local] = cached;
        let chunk_max = tree_max(local, score);
        let next_max = max(running_max, chunk_max);
        var p = 0.0;
        if (valid) {
            p = exp_refined(score - next_max);
        }
        shared_p[local] = p;
        let chunk_sum = tree_sum(local, p);
        let correction = exp_refined(running_max - next_max);
        running_sum = fma(running_sum, correction, chunk_sum);
        let members = min(WORKGROUP_SIZE, count - chunk * WORKGROUP_SIZE);
        var sum0 = 0.0;
        var sum1 = 0.0;
        for (var j = 0u; j < members; j = j + 1u) {
            let v_base = shared_token[j] * row_stride + kv_head * head_dim;
            if (dim0 < head_dim) {
                sum0 = fma(shared_p[j], f32(V[v_base + dim0]), sum0);
            }
            if (dim1 < head_dim) {
                sum1 = fma(shared_p[j], f32(V[v_base + dim1]), sum1);
            }
        }
        acc0 = fma(acc0, correction, sum0);
        acc1 = fma(acc1, correction, sum1);
        running_max = next_max;
        workgroupBarrier();
    }

    let out_base = (token * u.num_heads + head) * head_dim;
    if (dim0 < head_dim) {
        output[out_base + dim0] = select(0.0, acc0 / running_sum, running_sum > 0.0);
    }
    if (dim1 < head_dim) {
        output[out_base + dim1] = select(0.0, acc1 / running_sum, running_sum > 0.0);
    }
}
