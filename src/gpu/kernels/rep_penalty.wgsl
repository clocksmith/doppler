// Repetition Penalty Kernel
//
// Applies repetition penalty to logits in-place before sampling.
// Reads unique history token IDs (pre-deduplicated on CPU) and
// current-batch tokens from separate buffers.
//
// For each token ID: logit > 0 ? logit / penalty : logit * penalty
// Matches CPU applyRepetitionPenalty() behavior.
//
// One thread per token in (history + batch). Benign data race when the same
// token appears in both buffers: both threads compute identical results.

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    vocab_size: u32,
    history_count: u32,
    penalty: f32,
    batch_count: u32,
    batch_offset: u32,    // Offset into batch_tokens (skip startToken at index 0)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> logits: array<f32>;
@group(0) @binding(2) var<storage, read> history: array<u32>;
@group(0) @binding(3) var<storage, read> batch_tokens: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.history_count + u.batch_count;
    if (idx >= total) {
        return;
    }

    var token_id: u32;
    if (idx < u.history_count) {
        token_id = history[idx];
    } else {
        token_id = batch_tokens[u.batch_offset + (idx - u.history_count)];
    }

    if (token_id >= u.vocab_size) {
        return;
    }

    let penalty = u.penalty;
    let logit = logits[token_id];
    if (logit > 0.0) {
        logits[token_id] = logit / penalty;
    } else {
        logits[token_id] = logit * penalty;
    }
}
