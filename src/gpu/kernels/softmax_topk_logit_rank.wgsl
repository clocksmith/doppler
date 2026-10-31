// softmax_topk_logit_rank.wgsl
//
// Renormalized MoE top-k (normalize == 1) for one token per workgroup. Softmax is
// monotonic, so the experts rank by their logits (ties to the lower index), and the
// picked weights are the softmax over the K picked logits, exp(l_k - l_0) / sum_j
// exp(l_j - l_0), summed in rank order: the full softmax sum cancels in the
// renormalization. The selection equals softmax_topk_ranked.wgsl's except where exp()
// rounding ties two distinct logits; the weights differ from it in the last bits.
//
// Logits become order-preserving u32 keys (larger key = larger value; -0 keys as +0;
// NaN keys 0, behind every number; 0 also marks padding and taken entries). Stage 1:
// each subgroup takes its K best in order by K rounds of subgroupMax over the keys and
// subgroupMin over the indices holding the maximum. Stage 2: subgroup 0 takes the K
// best of those candidates the same way, each lane scanning the candidates it owns.
// Requires K <= subgroup size (checked by the wrapper).
enable subgroups;
requires subgroup_id;

override WORKGROUP_SIZE: u32 = 256u;
const MAX_EXPERTS: u32 = 256u;

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

var<workgroup> candidate_keys: array<u32, MAX_EXPERTS>;
var<workgroup> candidate_indices: array<u32, MAX_EXPERTS>;

fn order_key(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return 0u;
    }
    if (magnitude == 0u) {
        return 0x80000000u;
    }
    return select(~bits, bits | 0x80000000u, (bits & 0x80000000u) == 0u);
}

fn key_value(key: u32) -> f32 {
    return bitcast<f32>(select(~key, key & 0x7fffffffu, key >= 0x80000000u));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_index) t: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_id) subgroup: u32,
    @builtin(num_subgroups) subgroup_count: u32
) {
    let token_idx = wg_id.x;
    if (token_idx >= u.num_tokens) {
        return;
    }
    let n = u.num_experts;
    let top_k = u.top_k;
    var key = 0u;
    if (t < n) {
        key = order_key(probs[token_idx * n + t]);
    }

    // Stage 1: lane r of each subgroup keeps that subgroup's rank-r candidate.
    let per = min(top_k, subgroup_size);
    var own_key = 0u;
    var own_index = 0xffffffffu;
    for (var r = 0u; r < per; r++) {
        let best = subgroupMax(key);
        let winner = subgroupMin(select(0xffffffffu, t, key == best));
        if (lane == r) {
            own_key = best;
            own_index = winner;
        }
        if (t == winner) {
            key = 0u;
        }
    }
    if (lane < per) {
        candidate_keys[subgroup * per + lane] = own_key;
        candidate_indices[subgroup * per + lane] = own_index;
    }
    workgroupBarrier();
    if (subgroup != 0u) {
        return;
    }

    // Stage 2: candidates s with s % subgroup_size == lane belong to this lane, which
    // alone reads and clears them.
    let candidates = subgroup_count * per;
    var picked_key = 0u;
    var picked_index = 0xffffffffu;
    for (var r = 0u; r < top_k; r++) {
        var best_key = 0u;
        var best_index = 0xffffffffu;
        var best_slot = 0u;
        for (var s = lane; s < candidates; s += subgroup_size) {
            let candidate_key = candidate_keys[s];
            let candidate_index = candidate_indices[s];
            if (candidate_key > best_key || (candidate_key == best_key && candidate_index < best_index)) {
                best_key = candidate_key;
                best_index = candidate_index;
                best_slot = s;
            }
        }
        let best = subgroupMax(best_key);
        let winner = subgroupMin(select(0xffffffffu, best_index, best_key == best));
        if (lane == r) {
            picked_key = best;
            picked_index = winner;
        }
        if (best_key == best && best_index == winner) {
            candidate_keys[best_slot] = 0u;
            candidate_indices[best_slot] = 0xffffffffu;
        }
    }

    let value = key_value(picked_key);
    let e = exp(value - subgroupBroadcast(value, 0u));
    var weight_sum: f32 = 0.0;
    for (var k = 0u; k < top_k; k++) {
        weight_sum = weight_sum + subgroupShuffle(e, k);
    }
    if (lane < top_k) {
        out_indices[token_idx * top_k + lane] = picked_index;
        out_weights[token_idx * top_k + lane] = e * (1.0 / weight_sum);
    }
}
