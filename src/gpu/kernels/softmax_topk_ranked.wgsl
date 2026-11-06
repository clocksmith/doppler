// softmax_topk_ranked.wgsl
//
// Fused softmax + top-k for MoE routing with one invocation per expert.
// Arithmetic matches topk.wgsl#softmax_topk: the maximum is order independent,
// each exponent and probability is computed elementwise with the same
// expressions, the exponent sum and the top-k renormalization sum are
// accumulated serially in the same order by one invocation. Selection ranks
// every expert in parallel; ties order by the lower expert index.

override WORKGROUP_SIZE: u32 = 256u;
const MAX_EXPERTS: u32 = 256u;
const MAX_TOP_K: u32 = 64u;

struct Uniforms {
    num_tokens: u32,
    num_experts: u32,
    top_k: u32,
    normalize: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_weights: array<f32>;

var<workgroup> shared_values: array<f32, MAX_EXPERTS>;
var<workgroup> shared_scalar: f32;
var<workgroup> picked_probs: array<f32, MAX_TOP_K>;
var<workgroup> picked_indices: array<u32, MAX_TOP_K>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let t = local_id.x;
    let n = u.num_experts;
    let top_k = u.top_k;
    if (token_idx >= u.num_tokens) {
        return;
    }
    let is_expert = t < n;
    let base_offset = token_idx * n;
    var logit = 0.0;
    if (is_expert) {
        logit = probs[base_offset + t];
        shared_values[t] = logit;
    }
    workgroupBarrier();

    // Exact maximum: max is associative and commutative.
    if (t == 0u) {
        var max_val: f32 = shared_values[0];
        for (var i: u32 = 1u; i < min(n, 4u); i = i + 1u) {
            max_val = max(max_val, shared_values[i]);
        }
        for (var q: u32 = 1u; q < n / 4u; q = q + 1u) {
            let base = q * 4u;
            let c = vec4<f32>(shared_values[base], shared_values[base + 1u], shared_values[base + 2u], shared_values[base + 3u]);
            max_val = max(max(max(max(max_val, c.x), c.y), c.z), c.w);
        }
        for (var i: u32 = max(4u, n - n % 4u); i < n; i = i + 1u) {
            max_val = max(max_val, shared_values[i]);
        }
        shared_scalar = max_val;
    }
    workgroupBarrier();
    let max_val = shared_scalar;
    var exp_val = 0.0;
    if (is_expert) {
        exp_val = exp(logit - max_val);
    }
    workgroupBarrier();
    if (is_expert) {
        shared_values[t] = exp_val;
    }
    workgroupBarrier();

    // Same serial accumulation order as the scalar kernel.
    if (t == 0u) {
        var exp_sum: f32 = 0.0;
        for (var q: u32 = 0u; q < n / 4u; q = q + 1u) {
            let base = q * 4u;
            let c = vec4<f32>(shared_values[base], shared_values[base + 1u], shared_values[base + 2u], shared_values[base + 3u]);
            exp_sum = (((exp_sum + c.x) + c.y) + c.z) + c.w;
        }
        for (var i: u32 = n - n % 4u; i < n; i = i + 1u) {
            exp_sum = exp_sum + shared_values[i];
        }
        shared_scalar = 1.0 / exp_sum;
    }
    workgroupBarrier();
    let prob = exp_val * shared_scalar;
    workgroupBarrier();
    if (is_expert) {
        shared_values[t] = prob;
    }
    workgroupBarrier();

    if (is_expert) {
        var rank = 0u;
        let pv = vec4<f32>(prob);
        for (var q: u32 = 0u; q < (n + 3u) / 4u; q = q + 1u) {
            let base = q * 4u;
            let c = vec4<f32>(shared_values[base], shared_values[base + 1u], shared_values[base + 2u], shared_values[base + 3u]);
            let j = vec4<u32>(q * 4u) + vec4<u32>(0u, 1u, 2u, 3u);
            let beats = (c > pv) | ((c == pv) & (j < vec4<u32>(t)));
            let valid = j < vec4<u32>(n);
            rank = rank + dot(select(vec4<u32>(0u), vec4<u32>(1u), beats & valid), vec4<u32>(1u));
        }
        if (rank < top_k) {
            picked_probs[rank] = prob;
            picked_indices[rank] = t;
        }
    }
    workgroupBarrier();

    if (t == 0u) {
        var weight_sum: f32 = 0.0;
        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            weight_sum = weight_sum + picked_probs[k];
        }
        shared_scalar = select(1.0, 1.0 / weight_sum, u.normalize == 1u && weight_sum > 0.0);
    }
    workgroupBarrier();
    if (t < top_k) {
        let out_base = token_idx * top_k;
        out_indices[out_base + t] = picked_indices[t];
        out_weights[out_base + t] = picked_probs[t] * shared_scalar;
    }
}
