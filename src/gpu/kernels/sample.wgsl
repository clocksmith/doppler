// sample.wgsl

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
    if (candidate_index == 0xffffffffu) { return false; }
    if (best_index == 0xffffffffu) { return true; }
    if (candidate_value > best_value) {
        return true;
    }
    if (candidate_value < best_value) {
        return false;
    }
    return candidate_index < best_index;
}

fn finite_candidate(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;              // [vocabSize]
@group(0) @binding(2) var<storage, read_write> output: array<u32>;         // [N] - selected tokens
@group(0) @binding(3) var<storage, read_write> topk_indices: array<atomic<u32>>;    // [topK] - intermediate
@group(0) @binding(4) var<storage, read_write> topk_logits: array<atomic<u32>>;     // [topK] - intermediate


// Explicit scratch loads/stores preserve heap mutations across Metal compilers.
// Scores retain their F32 bits; indices and buffer sizes are unchanged.
fn load_score(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&topk_logits[i])); }
fn store_score(i: u32, value: f32) { atomicStore(&topk_logits[i], bitcast<u32>(value)); }

fn load_index(i: u32) -> u32 { return atomicLoad(&topk_indices[i]); }
fn store_index(i: u32, value: u32) { atomicStore(&topk_indices[i], value); }

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
    let value = load_score(base + a);
    let token = load_index(base + a);
    store_score(base + a, load_score(base + b));
    store_index(base + a, load_index(base + b));
    store_score(base + b, value);
    store_index(base + b, token);
}

fn heap_better(base: u32, a: u32, b: u32) -> bool {
    return candidate_beats(load_score(base + a), load_index(base + a),
        load_score(base + b), load_index(base + b));
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
        store_score(base + count, value);
        store_index(base + count, token);
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
    if (candidate_beats(value, token, load_score(base), load_index(base))) {
        store_score(base, value);
        store_index(base, token);
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
    for (var k = 0u; k < u.top_k; k++) { store_index(base + k, 0xffffffffu); }
    var count = 0u;
    for (var token = wgid.x; token < u.vocab_size; token += num_wg.x) {
        let raw = logits[token];
        // Non-finite and padding logits never enter the candidate distribution.
        if (token != u.pad_token_id && finite_candidate(raw)) {
            let value = apply_softcap(raw, u.logit_softcap);
            count = heap_offer(base, count, value, token);
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn find_topk_phase2(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x != 0u) { return; }
    let base = sample_group_count() * u.top_k;
    for (var k = 0u; k < u.top_k; k++) { store_index(base + k, 0xffffffffu); }
    var count = 0u;
    for (var i = 0u; i < base; i++) {
        if (load_index(i) != 0xffffffffu) {
            count = heap_offer(base, count, load_score(i), load_index(i));
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
    while (count < u.top_k && load_index(base + count) != 0xffffffffu) { count++; }
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
    let maximum = load_score(base);
    var total = 0.0;
    for (var i = 0u; i < count; i++) {
        let weight = exp((load_score(base + i) - maximum) / u.temperature);
        store_score(base + i, weight);
        total += weight;
    }
    var retained = 0u;
    var retained_total = 0.0;
    loop {
        retained_total += load_score(base + retained);
        retained++;
        if (retained >= count || retained_total >= u.top_p * total) { break; }
    }
    let threshold = u.random_value * retained_total;
    var cumulative = 0.0;
    var selected = load_index(base + retained - 1u);
    for (var i = 0u; i < retained; i++) {
        cumulative += load_score(base + i);
        if (cumulative > threshold) {
            selected = load_index(base + i);
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
    var local_max_idx: u32 = 0xffffffffu;

    var idx = gid.x;
    while (idx < vocab_size) {
        if (idx != pad_id && finite_candidate(f32(logits[idx]))) {
            // Apply softcapping before temperature scaling
            let val = apply_softcap(logits[idx], softcap) / temperature;
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
    var local_max_idx: u32 = 0xffffffffu;

    var idx = global_idx;
    while (idx < vocab_size) {
        if (idx != pad_id && finite_candidate(f32(logits[idx]))) {
            // Apply softcapping (argmax is greedy, no temperature)
            let val = apply_softcap(logits[idx], softcap);
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
        store_score(wgid.x, shared_values[0]);
        store_index(wgid.x, shared_indices[0]);
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
        shared_values[thread_idx] = load_score(thread_idx);
        shared_indices[thread_idx] = load_index(thread_idx);
    } else {
        shared_values[thread_idx] = NEG_INF;
        shared_indices[thread_idx] = 0xffffffffu;
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
