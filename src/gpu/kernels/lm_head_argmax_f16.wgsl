// lm-head-argmax-f16.wgsl

enable f16;

override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 64u;
override THREADS_PER_COL: u32 = 4u;

const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_COLS_PER_WG: u32 = 64u;
const NEG_INF: f32 = -3.402823e+38;

struct Uniforms {
    vocab_size: u32,
    hidden_size: u32,
    transpose_b: u32,
    workgroups_x: u32,
    pad_token_id: u32,
    logit_softcap: f32,
    output_index: u32,
    num_groups: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> hidden: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read_write> temp_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> temp_logits: array<f32>;

var<workgroup> partial_sums: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> candidate_values: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> candidate_indices: array<u32, MAX_WORKGROUP_SIZE>;

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

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn phase1(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let local_id = lid.x;
    let col_in_wg = local_id / THREADS_PER_COL;
    let thread_in_col = local_id % THREADS_PER_COL;
    let wg_linear = wg_id.y * u.workgroups_x + wg_id.x;
    let base_col = wg_linear * COLS_PER_WG;
    let col = base_col + col_in_wg;
    let is_valid = col < u.vocab_size && col != u.pad_token_id;

    var partial_sum: f32 = 0.0;
    if (is_valid) {
        let k_per_thread = (u.hidden_size + THREADS_PER_COL - 1u) / THREADS_PER_COL;
        let k_start = thread_in_col * k_per_thread;
        let k_end = min(k_start + k_per_thread, u.hidden_size);
        var k = k_start;
        let k_aligned_end = k_start + ((k_end - k_start) / 4u) * 4u;

        if (u.transpose_b == 1u) {
            let row_offset = col * u.hidden_size;
            for (; k < k_aligned_end; k = k + 4u) {
                let a = vec4<f32>(hidden[k], hidden[k + 1u], hidden[k + 2u], hidden[k + 3u]);
                let b = vec4<f32>(
                    f32(weights[row_offset + k]),
                    f32(weights[row_offset + k + 1u]),
                    f32(weights[row_offset + k + 2u]),
                    f32(weights[row_offset + k + 3u])
                );
                partial_sum = partial_sum + dot(a, b);
            }
            for (; k < k_end; k = k + 1u) {
                partial_sum = partial_sum + hidden[k] * f32(weights[row_offset + k]);
            }
        } else {
            for (; k < k_aligned_end; k = k + 4u) {
                let a = vec4<f32>(hidden[k], hidden[k + 1u], hidden[k + 2u], hidden[k + 3u]);
                let b = vec4<f32>(
                    f32(weights[k * u.vocab_size + col]),
                    f32(weights[(k + 1u) * u.vocab_size + col]),
                    f32(weights[(k + 2u) * u.vocab_size + col]),
                    f32(weights[(k + 3u) * u.vocab_size + col])
                );
                partial_sum = partial_sum + dot(a, b);
            }
            for (; k < k_end; k = k + 1u) {
                partial_sum = partial_sum + hidden[k] * f32(weights[k * u.vocab_size + col]);
            }
        }
    }

    partial_sums[local_id] = partial_sum;
    workgroupBarrier();

    if (thread_in_col == 0u && col_in_wg < MAX_COLS_PER_WG) {
        let base = col_in_wg * THREADS_PER_COL;
        var sum = partial_sums[base];
        for (var i: u32 = 1u; i < THREADS_PER_COL; i = i + 1u) {
            sum = sum + partial_sums[base + i];
        }
        candidate_values[col_in_wg] = select(NEG_INF, apply_softcap(sum, u.logit_softcap), is_valid);
        candidate_indices[col_in_wg] = col;
    }
    workgroupBarrier();

    if (local_id < COLS_PER_WG) {
        candidate_values[local_id] = select(NEG_INF, candidate_values[local_id], local_id < MAX_COLS_PER_WG);
        candidate_indices[local_id] = select(0u, candidate_indices[local_id], local_id < MAX_COLS_PER_WG);
    }
    workgroupBarrier();

    var stride = COLS_PER_WG / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            if (candidate_beats(
                candidate_values[local_id + stride],
                candidate_indices[local_id + stride],
                candidate_values[local_id],
                candidate_indices[local_id]
            )) {
                candidate_values[local_id] = candidate_values[local_id + stride];
                candidate_indices[local_id] = candidate_indices[local_id + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_id == 0u) {
        temp_logits[wg_linear] = candidate_values[0];
        temp_indices[wg_linear] = candidate_indices[0];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn phase2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local_id = lid.x;
    var local_best_value: f32 = NEG_INF;
    var local_best_index: u32 = 0u;

    var group = local_id;
    while (group < u.num_groups) {
        let candidate_value = temp_logits[group];
        let candidate_index = temp_indices[group];
        if (candidate_beats(candidate_value, candidate_index, local_best_value, local_best_index)) {
            local_best_value = candidate_value;
            local_best_index = candidate_index;
        }
        group = group + WORKGROUP_SIZE;
    }

    candidate_values[local_id] = local_best_value;
    candidate_indices[local_id] = local_best_index;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            if (candidate_beats(
                candidate_values[local_id + stride],
                candidate_indices[local_id + stride],
                candidate_values[local_id],
                candidate_indices[local_id]
            )) {
                candidate_values[local_id] = candidate_values[local_id + stride];
                candidate_indices[local_id] = candidate_indices[local_id + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_id == 0u) {
        output[u.output_index] = candidate_indices[0];
    }
}
