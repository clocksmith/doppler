// Strided scalar ranking for banks larger than one workgroup. Unlike the
// vector-ranked path, scalar workgroup elements have independent write owners.
override WORKGROUP_SIZE: u32 = 256u;
override MAX_EXPERTS: u32 = 512u;
const MAX_TOP_K: u32 = 64u;

struct Uniforms {
    num_tokens: u32,
    num_experts: u32,
    top_k: u32,
    normalize: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_weights: array<f32>;

var<workgroup> values: array<f32, MAX_EXPERTS>;
var<workgroup> shared_scalar: f32;
var<workgroup> picked_probs: array<f32, MAX_TOP_K>;
var<workgroup> picked_indices: array<u32, MAX_TOP_K>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let token = workgroup_id.x;
    let lane = local_id.x;
    let n = u.num_experts;
    if (token >= u.num_tokens) { return; }
    let base = token * n;
    for (var expert = lane; expert < n; expert += WORKGROUP_SIZE) {
        values[expert] = logits[base + expert];
    }
    workgroupBarrier();
    if (lane == 0u) {
        var maximum = values[0];
        for (var expert = 1u; expert < n; expert++) { maximum = max(maximum, values[expert]); }
        shared_scalar = maximum;
    }
    workgroupBarrier();
    let maximum = shared_scalar;
    for (var expert = lane; expert < n; expert += WORKGROUP_SIZE) {
        values[expert] = exp(logits[base + expert] - maximum);
    }
    workgroupBarrier();
    if (lane == 0u) {
        var total = 0.0;
        for (var expert = 0u; expert < n; expert++) { total += values[expert]; }
        shared_scalar = 1.0 / total;
    }
    workgroupBarrier();
    let inverse_total = shared_scalar;
    for (var expert = lane; expert < n; expert += WORKGROUP_SIZE) {
        values[expert] = exp(logits[base + expert] - maximum) * inverse_total;
    }
    workgroupBarrier();
    for (var expert = lane; expert < n; expert += WORKGROUP_SIZE) {
        let probability = values[expert];
        var rank = 0u;
        for (var other = 0u; other < n; other++) {
            if (values[other] > probability || (values[other] == probability && other < expert)) { rank++; }
        }
        if (rank < u.top_k) {
            picked_probs[rank] = probability;
            picked_indices[rank] = expert;
        }
    }
    workgroupBarrier();
    if (lane == 0u) {
        var total = 0.0;
        for (var pick = 0u; pick < u.top_k; pick++) { total += picked_probs[pick]; }
        shared_scalar = select(1.0, 1.0 / total, u.normalize == 1u && total > 0.0);
    }
    workgroupBarrier();
    if (lane < u.top_k) {
        out_indices[token * u.top_k + lane] = picked_indices[lane];
        out_weights[token * u.top_k + lane] = picked_probs[lane] * shared_scalar;
    }
}
