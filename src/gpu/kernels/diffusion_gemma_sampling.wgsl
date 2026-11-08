override WORKGROUP_SIZE: u32 = 256u;

const NEG_INF: f32 = -3.402823e+38;
const NO_PAD_TOKEN: u32 = 0xffffffffu;

struct Uniforms {
    vocab_size: u32,
    canvas_length: u32,
    pad_token_id: u32,
    _pad0: u32,
    temperature: f32,
    logit_softcap: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> argmax_tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> entropies: array<f32>;
// One uniform draw in [0, 1) per canvas position, supplied by the caller so the
// GPU path consumes the same random stream as the CPU reference.
@group(0) @binding(4) var<storage, read> uniform_draws: array<f32>;
@group(0) @binding(5) var<storage, read_write> sampled_tokens: array<u32>;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_indices: array<u32, 256>;
var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_weighted: array<f32, 256>;
var<workgroup> shared_block_sum: array<f32, 256>;

fn apply_softcap(x: f32, softcap: f32) -> f32 {
    if (softcap <= 0.0) {
        return x;
    }
    return softcap * tanh(x / softcap);
}

fn candidate_beats(candidate_value: f32, candidate_index: u32, best_value: f32, best_index: u32) -> bool {
    if (candidate_value > best_value) {
        return true;
    }
    if (candidate_value < best_value) {
        return false;
    }
    return candidate_index < best_index;
}

fn is_sampleable(idx: u32, pad_id: u32) -> bool {
    return pad_id == NO_PAD_TOKEN || idx != pad_id;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn entropy_stats(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = workgroup_id.x;
    let thread_idx = local_id.x;
    if (row >= u.canvas_length) {
        return;
    }

    let base = row * u.vocab_size;
    let temperature = max(u.temperature, 0.000001);
    let pad_id = u.pad_token_id;
    let softcap = u.logit_softcap;

    var local_max = NEG_INF;
    var local_index = 0u;
    var idx = thread_idx;
    while (idx < u.vocab_size) {
        if (is_sampleable(idx, pad_id)) {
            let value = apply_softcap(logits[base + idx], softcap) / temperature;
            if (candidate_beats(value, idx, local_max, local_index)) {
                local_max = value;
                local_index = idx;
            }
        }
        idx = idx + WORKGROUP_SIZE;
    }

    shared_max[thread_idx] = local_max;
    shared_indices[thread_idx] = local_index;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            if (candidate_beats(
                shared_max[thread_idx + stride],
                shared_indices[thread_idx + stride],
                shared_max[thread_idx],
                shared_indices[thread_idx]
            )) {
                shared_max[thread_idx] = shared_max[thread_idx + stride];
                shared_indices[thread_idx] = shared_indices[thread_idx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let row_max = shared_max[0];

    var local_sum = 0.0;
    var local_weighted = 0.0;
    idx = thread_idx;
    while (idx < u.vocab_size) {
        if (is_sampleable(idx, pad_id)) {
            let value = apply_softcap(logits[base + idx], softcap) / temperature;
            let exp_value = exp(value - row_max);
            local_sum = local_sum + exp_value;
            local_weighted = local_weighted + exp_value * value;
        }
        idx = idx + WORKGROUP_SIZE;
    }

    shared_sum[thread_idx] = local_sum;
    shared_weighted[thread_idx] = local_weighted;
    workgroupBarrier();

    stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
            shared_weighted[thread_idx] = shared_weighted[thread_idx] + shared_weighted[thread_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Inverse-CDF sample in vocabulary order. Each thread sums one contiguous
    // block of the vocabulary; thread 0 walks the block sums to the block that
    // holds the target mass and then walks that block element by element. The
    // first token whose cumulative mass reaches the target is the sample, the
    // same rule the CPU reference applies.
    let block_size = (u.vocab_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let block_start = thread_idx * block_size;
    let block_end = min(block_start + block_size, u.vocab_size);
    var block_sum = 0.0;
    var j = block_start;
    while (j < block_end) {
        if (is_sampleable(j, pad_id)) {
            let value = apply_softcap(logits[base + j], softcap) / temperature;
            block_sum = block_sum + exp(value - row_max);
        }
        j = j + 1u;
    }
    shared_block_sum[thread_idx] = block_sum;
    workgroupBarrier();

    if (thread_idx == 0u) {
        var total = 0.0;
        for (var b = 0u; b < WORKGROUP_SIZE; b = b + 1u) {
            total = total + shared_block_sum[b];
        }
        let target_mass = uniform_draws[row] * total;
        var cumulative = 0.0;
        var sampled = u.vocab_size - 1u;
        if (sampled == pad_id && sampled > 0u) {
            sampled = sampled - 1u;
        }
        var found = false;
        var b = 0u;
        while (b < WORKGROUP_SIZE && !found) {
            let block_total = cumulative + shared_block_sum[b];
            if (target_mass <= block_total && shared_block_sum[b] > 0.0) {
                var k = b * block_size;
                let k_end = min(k + block_size, u.vocab_size);
                while (k < k_end && !found) {
                    if (is_sampleable(k, pad_id)) {
                        let value = apply_softcap(logits[base + k], softcap) / temperature;
                        cumulative = cumulative + exp(value - row_max);
                        if (target_mass <= cumulative) {
                            sampled = k;
                            found = true;
                        }
                    }
                    k = k + 1u;
                }
            } else {
                cumulative = block_total;
            }
            b = b + 1u;
        }

        let sum = shared_sum[0];
        let entropy = select(0.0, log(sum) + row_max - (shared_weighted[0] / sum), sum > 0.0);
        argmax_tokens[row] = shared_indices[0];
        entropies[row] = entropy;
        sampled_tokens[row] = sampled;
    }
}
