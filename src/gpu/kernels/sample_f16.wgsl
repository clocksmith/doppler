// AUTO-GENERATED from src/gpu/kernels/sample.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// sample_f16.wgsl: F16 storage, F32 sampling accumulation.

enable f16;

/**
 * GPU-Side Sampling Kernel
 *
 * Performs temperature scaling, top-k selection, softmax, and sampling
 * entirely on GPU. Only reads back the single selected token ID.
 *
 * Reduces readback from 1MB (256K vocab × 4 bytes) to 4 bytes.
 *
 * Algorithm:
 * 1. Temperature scaling: logits = logits / temperature
 * 2. Parallel top-k: Each workgroup finds local top-k, then merge
 * 3. Softmax on top-k candidates
 * 4. Multinomial sampling with provided random value
 */

// Configuration
override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.402823e+38;

struct Uniforms {
    vocab_size: u32,
    top_k: u32,
    temperature: f32,
    random_value: f32,  // Pre-generated random [0,1) for sampling
    pad_token_id: u32,
    logit_softcap: f32,  // Gemma 2: 30.0, 0.0 = disabled
    output_index: u32,   // Index into output token buffer
    top_p: f32,
}

// Apply softcapping: softcap * tanh(x / softcap)
// Returns x unchanged if softcap <= 0
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

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> logits: array<f16>;              // [vocabSize]
@group(0) @binding(2) var<storage, read_write> output: array<u32>;         // [N] - selected tokens
@group(0) @binding(3) var<storage, read_write> topk_indices: array<u32>;    // [topK] - intermediate
@group(0) @binding(4) var<storage, read_write> topk_logits: array<f32>;     // [topK] - intermediate

// Shared memory for workgroup-level reduction
var<workgroup> shared_values: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_indices: array<u32, MAX_WORKGROUP_SIZE>;

// Each partition retains its exact top-k in a min-heap. The merge has a
// separate output region, so retaining a winner cannot overwrite unread input.
// Scratch holds at most two vocabularies; top_k is resolved to [1, vocab_size].
fn sample_group_count() -> u32 {
    return max(1u, min(min(WORKGROUP_SIZE, (u.vocab_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE),
        u.vocab_size / u.top_k));
}

fn heap_swap(base: u32, a: u32, b: u32) {
    let value = topk_logits[base + a];
    let token = topk_indices[base + a];
    topk_logits[base + a] = topk_logits[base + b];
    topk_indices[base + a] = topk_indices[base + b];
    topk_logits[base + b] = value;
    topk_indices[base + b] = token;
}

fn heap_better(base: u32, a: u32, b: u32) -> bool {
    return candidate_beats(topk_logits[base + a], topk_indices[base + a],
        topk_logits[base + b], topk_indices[base + b]);
}

fn heap_down(base: u32, count: u32) {
    var root = 0u;
    loop {
        let left = root * 2u + 1u;
        if (left >= count) { break; }
        var worst = left;
        let right = left + 1u;
        if (right < count && heap_better(base, left, right)) { worst = right; }
        if (!heap_better(base, root, worst)) { break; }
        heap_swap(base, root, worst);
        root = worst;
    }
}

fn heap_offer(base: u32, count: u32, value: f32, token: u32) -> u32 {
    if (count < u.top_k) {
        topk_logits[base + count] = value;
        topk_indices[base + count] = token;
        var child = count;
        loop {
            if (child == 0u) { break; }
            let parent = (child - 1u) / 2u;
            if (!heap_better(base, parent, child)) { break; }
            heap_swap(base, parent, child);
            child = parent;
        }
        return count + 1u;
    }
    if (candidate_beats(value, token, topk_logits[base], topk_indices[base])) {
        topk_logits[base] = value;
        topk_indices[base] = token;
        heap_down(base, count);
    }
    return count;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn find_topk_phase1(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    if (lid.x != 0u) { return; }
    let base = wgid.x * u.top_k;
    for (var k = 0u; k < u.top_k; k++) { topk_indices[base + k] = 0xffffffffu; }
    var count = 0u;
    for (var token = wgid.x; token < u.vocab_size; token += num_wg.x) {
        let raw = f32(logits[token]);
        // Non-finite and padding logits never enter the candidate distribution.
        if (token != u.pad_token_id && abs(raw) <= 3.402823e+38) {
            let value = apply_softcap(raw, u.logit_softcap);
            count = heap_offer(base, count, value, token);
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn find_topk_phase2(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x != 0u) { return; }
    let base = sample_group_count() * u.top_k;
    for (var k = 0u; k < u.top_k; k++) { topk_indices[base + k] = 0xffffffffu; }
    var count = 0u;
    for (var i = 0u; i < base; i++) {
        if (topk_indices[i] != 0xffffffffu) {
            count = heap_offer(base, count, topk_logits[i], topk_indices[i]);
        }
    }
}

// Sort the retained heap, normalize top-k, retain the minimal top-p prefix,
// then sample from that renormalized prefix. F16 inputs use F32 accumulation.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn softmax_and_sample(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x != 0u) { return; }
    let base = sample_group_count() * u.top_k;
    var count = 0u;
    while (count < u.top_k && topk_indices[base + count] != 0xffffffffu) { count++; }
    if (count == 0u) {
        output[u.output_index] = 0xffffffffu;
        return;
    }
    var remaining = count;
    while (remaining > 1u) {
        remaining--;
        heap_swap(base, 0u, remaining);
        heap_down(base, remaining);
    }
    let maximum = topk_logits[base];
    var total = 0.0;
    for (var i = 0u; i < count; i++) {
        let weight = exp((topk_logits[base + i] - maximum) / u.temperature);
        topk_logits[base + i] = weight;
        total += weight;
    }
    var retained = 0u;
    var retained_total = 0.0;
    loop {
        retained_total += topk_logits[base + retained];
        retained++;
        if (retained >= count || retained_total >= u.top_p * total) { break; }
    }
    let threshold = u.random_value * retained_total;
    var cumulative = 0.0;
    var selected = topk_indices[base + retained - 1u];
    for (var i = 0u; i < retained; i++) {
        cumulative += topk_logits[base + i];
        if (cumulative >= threshold) {
            selected = topk_indices[base + i];
            break;
        }
    }
    output[u.output_index] = selected;
}

// Combined single-pass version for smaller vocabularies (<= 65536)
// Uses hierarchical reduction within single kernel
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn sample_single_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let thread_idx = lid.x;
    let vocab_size = u.vocab_size;
    let temperature = u.temperature;
    let random_val = u.random_value;
    let pad_id = u.pad_token_id;
    let softcap = u.logit_softcap;

    // Phase 1: Find global max
    var local_max: f32 = NEG_INF;
    var local_max_idx: u32 = 0u;

    var idx = gid.x;
    while (idx < vocab_size) {
        if (idx != pad_id) {
            // Apply softcapping before temperature scaling
            let val = apply_softcap(f32(logits[idx]), softcap) / temperature;
            if (candidate_beats(val, idx, local_max, local_max_idx)) {
                local_max = val;
                local_max_idx = idx;
            }
        }
        idx = idx + num_wg.x * WORKGROUP_SIZE;
    }

    shared_values[thread_idx] = local_max;
    shared_indices[thread_idx] = local_max_idx;
    workgroupBarrier();

    // Reduce to find workgroup max
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            if (candidate_beats(
                shared_values[thread_idx + stride],
                shared_indices[thread_idx + stride],
                shared_values[thread_idx],
                shared_indices[thread_idx]
            )) {
                shared_values[thread_idx] = shared_values[thread_idx + stride];
                shared_indices[thread_idx] = shared_indices[thread_idx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // For single workgroup, thread 0 can do everything
    if (thread_idx == 0u && num_wg.x == 1u) {
        // We have top-1, but need top-k
        // For small vocab, just do the full selection
        // This simplified version selects top-1 only (greedy)
        // Full top-k sampling requires multi-pass for large vocab

        output[u.output_index] = shared_indices[0];
    }
}

// Greedy argmax for deterministic decoding (temperature=0 equivalent)
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

    // Each thread finds max in its chunk
    var local_max: f32 = NEG_INF;
    var local_max_idx: u32 = 0u;

    var idx = global_idx;
    while (idx < vocab_size) {
        if (idx != pad_id) {
            // Apply softcapping (argmax is greedy, no temperature)
            let val = apply_softcap(f32(logits[idx]), softcap);
            if (candidate_beats(val, idx, local_max, local_max_idx)) {
                local_max = val;
                local_max_idx = idx;
            }
        }
        idx = idx + num_wg.x * WORKGROUP_SIZE;
    }

    shared_values[thread_idx] = local_max;
    shared_indices[thread_idx] = local_max_idx;
    workgroupBarrier();

    // Reduce within workgroup
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            if (candidate_beats(
                shared_values[thread_idx + stride],
                shared_indices[thread_idx + stride],
                shared_values[thread_idx],
                shared_indices[thread_idx]
            )) {
                shared_values[thread_idx] = shared_values[thread_idx + stride];
                shared_indices[thread_idx] = shared_indices[thread_idx + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Write workgroup result to global memory
    if (thread_idx == 0u) {
        topk_logits[wgid.x] = shared_values[0];
        topk_indices[wgid.x] = shared_indices[0];
    }
}

// Final reduction for argmax across workgroups
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn argmax_reduce(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let thread_idx = lid.x;
    let num_groups = min(WORKGROUP_SIZE, (u.vocab_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE);

    // Load workgroup maxes (up to WORKGROUP_SIZE)
    if (thread_idx < num_groups) {
        shared_values[thread_idx] = topk_logits[thread_idx];
        shared_indices[thread_idx] = topk_indices[thread_idx];
    } else {
        shared_values[thread_idx] = NEG_INF;
        shared_indices[thread_idx] = 0u;
    }
    workgroupBarrier();

    // Reduce
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (thread_idx < stride) {
            if (candidate_beats(
                shared_values[thread_idx + stride],
                shared_indices[thread_idx + stride],
                shared_values[thread_idx],
                shared_indices[thread_idx]
            )) {
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
