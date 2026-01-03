// sample_f16.wgsl

/**
 * GPU-Side Sampling Kernel (f16 logits)
 *
 * Same as sample.wgsl but logits are f16.
 * Computation is performed in f32 for stability.
 */

enable f16;

// Configuration
override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_TOP_K: u32 = 128u;

struct Uniforms {
    vocab_size: u32,
    top_k: u32,
    temperature: f32,
    random_value: f32,
    pad_token_id: u32,
    logit_softcap: f32,
    output_index: u32,
    pad0: u32,
}

fn apply_softcap(x: f32, softcap: f32) -> f32 {
    if (softcap <= 0.0) {
        return x;
    }
    return softcap * tanh(x / softcap);
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> logits: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> topk_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> topk_logits: array<f32>;

var<workgroup> shared_values: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_indices: array<u32, MAX_WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn find_topk_phase1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let thread_idx = lid.x;
    let global_idx = gid.x;
    let vocab_size = u.vocab_size;
    let temperature = u.temperature;
    let pad_id = u.pad_token_id;
    let softcap = u.logit_softcap;

    if (num_wg.x == 1u) {
        var val: f32 = -3.402823e+38;
        if (global_idx < vocab_size && global_idx != pad_id) {
            val = apply_softcap(f32(logits[global_idx]), softcap) / temperature;
        }
        topk_logits[thread_idx] = val;
        topk_indices[thread_idx] = global_idx;
        return;
    }

    var local_max: f32 = -3.402823e+38;
    var local_max_idx: u32 = 0u;

    var idx = global_idx;
    while (idx < vocab_size) {
        if (idx != pad_id) {
            let val = apply_softcap(f32(logits[idx]), softcap) / temperature;
            if (val > local_max) {
                local_max = val;
                local_max_idx = idx;
            }
        }
        idx = idx + WORKGROUP_SIZE * num_wg.x;
    }

    shared_values[thread_idx] = local_max;
    shared_indices[thread_idx] = local_max_idx;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            if (shared_values[thread_idx + stride] > shared_values[thread_idx]) {
                shared_values[thread_idx] = shared_values[thread_idx + stride];
                shared_indices[thread_idx] = shared_indices[thread_idx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (thread_idx == 0u) {
        topk_logits[wgid.x] = shared_values[0];
        topk_indices[wgid.x] = shared_indices[0];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn find_topk_phase2(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let thread_idx = lid.x;
    let top_k = u.top_k;
    let num_groups = min(WORKGROUP_SIZE, (u.vocab_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE);
    let num_candidates = select(num_groups, min(u.vocab_size, WORKGROUP_SIZE), num_groups == 1u);

    if (thread_idx < WORKGROUP_SIZE) {
        if (thread_idx < num_candidates) {
            shared_values[thread_idx] = topk_logits[thread_idx];
            shared_indices[thread_idx] = topk_indices[thread_idx];
        } else {
            shared_values[thread_idx] = -3.402823e+38;
            shared_indices[thread_idx] = 0u;
        }
    }
    workgroupBarrier();

    if (thread_idx == 0u) {
        for (var k: u32 = 0u; k < top_k && k < num_candidates; k = k + 1u) {
            var max_idx = k;
            var max_val = shared_values[k];

            for (var i: u32 = k + 1u; i < num_candidates; i = i + 1u) {
                if (shared_values[i] > max_val) {
                    max_val = shared_values[i];
                    max_idx = i;
                }
            }

            if (max_idx != k) {
                let tmp_val = shared_values[k];
                let tmp_idx = shared_indices[k];
                shared_values[k] = shared_values[max_idx];
                shared_indices[k] = shared_indices[max_idx];
                shared_values[max_idx] = tmp_val;
                shared_indices[max_idx] = tmp_idx;
            }
        }

        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            topk_logits[k] = shared_values[k];
            topk_indices[k] = shared_indices[k];
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn softmax_and_sample(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let thread_idx = lid.x;
    let top_k = u.top_k;
    let random_val = u.random_value;

    if (thread_idx < top_k) {
        shared_values[thread_idx] = topk_logits[thread_idx];
        shared_indices[thread_idx] = topk_indices[thread_idx];
    }
    workgroupBarrier();

    if (thread_idx == 0u) {
        var max_val: f32 = shared_values[0];
        for (var i: u32 = 1u; i < top_k; i = i + 1u) {
            max_val = max(max_val, shared_values[i]);
        }

        var exp_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < top_k; i = i + 1u) {
            let exp_val = exp(shared_values[i] - max_val);
            shared_values[i] = exp_val;
            exp_sum = exp_sum + exp_val;
        }

        let inv_sum = 1.0 / exp_sum;
        var cum_prob: f32 = 0.0;
        var selected_token: u32 = shared_indices[top_k - 1u];

        for (var i: u32 = 0u; i < top_k; i = i + 1u) {
            let prob = shared_values[i] * inv_sum;
            cum_prob = cum_prob + prob;
            if (cum_prob >= random_val) {
                selected_token = shared_indices[i];
                break;
            }
        }

        output[u.output_index] = selected_token;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn sample_single_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let thread_idx = lid.x;
    let vocab_size = u.vocab_size;
    let top_k = min(u.top_k, MAX_TOP_K);
    let temperature = u.temperature;
    let random_val = u.random_value;
    let pad_id = u.pad_token_id;
    let softcap = u.logit_softcap;

    var local_max: f32 = -3.402823e+38;
    var local_max_idx: u32 = 0u;

    var idx = gid.x;
    while (idx < vocab_size) {
        if (idx != pad_id) {
            let val = apply_softcap(f32(logits[idx]), softcap) / temperature;
            if (val > local_max) {
                local_max = val;
                local_max_idx = idx;
            }
        }
        idx = idx + num_wg.x * WORKGROUP_SIZE;
    }

    shared_values[thread_idx] = local_max;
    shared_indices[thread_idx] = local_max_idx;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            if (shared_values[thread_idx + stride] > shared_values[thread_idx]) {
                shared_values[thread_idx] = shared_values[thread_idx + stride];
                shared_indices[thread_idx] = shared_indices[thread_idx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (thread_idx == 0u && num_wg.x == 1u) {
        output[u.output_index] = shared_indices[0];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn argmax(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let thread_idx = lid.x;
    let global_idx = gid.x;
    let vocab_size = u.vocab_size;
    let pad_id = u.pad_token_id;
    let softcap = u.logit_softcap;

    var local_max: f32 = -3.402823e+38;
    var local_max_idx: u32 = 0u;

    var idx = global_idx;
    while (idx < vocab_size) {
        if (idx != pad_id) {
            let val = apply_softcap(f32(logits[idx]), softcap);
            if (val > local_max) {
                local_max = val;
                local_max_idx = idx;
            }
        }
        idx = idx + num_wg.x * WORKGROUP_SIZE;
    }

    shared_values[thread_idx] = local_max;
    shared_indices[thread_idx] = local_max_idx;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            if (shared_values[thread_idx + stride] > shared_values[thread_idx]) {
                shared_values[thread_idx] = shared_values[thread_idx + stride];
                shared_indices[thread_idx] = shared_indices[thread_idx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (thread_idx == 0u) {
        topk_logits[wgid.x] = shared_values[0];
        topk_indices[wgid.x] = shared_indices[0];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn argmax_reduce(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let thread_idx = lid.x;
    let num_groups = min(WORKGROUP_SIZE, (u.vocab_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE);

    if (thread_idx < num_groups) {
        shared_values[thread_idx] = topk_logits[thread_idx];
        shared_indices[thread_idx] = topk_indices[thread_idx];
    } else {
        shared_values[thread_idx] = -3.402823e+38;
        shared_indices[thread_idx] = 0u;
    }
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            if (shared_values[thread_idx + stride] > shared_values[thread_idx]) {
                shared_values[thread_idx] = shared_values[thread_idx + stride];
                shared_indices[thread_idx] = shared_indices[thread_idx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (thread_idx == 0u) {
        output[u.output_index] = shared_indices[0];
    }
}