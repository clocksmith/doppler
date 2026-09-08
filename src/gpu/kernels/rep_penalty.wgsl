// Repetition followed by presence penalty over the ordered recent context.
// Only the first occurrence inside the window writes a token's logit.

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    vocab_size: u32,
    history_count: u32,
    penalty: f32,
    batch_count: u32,
    batch_offset: u32,
    presence_penalty: f32,
    repetition_penalty_window: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> logits: array<f32>;
@group(0) @binding(2) var<storage, read> history: array<u32>;
@group(0) @binding(3) var<storage, read> batch_tokens: array<u32>;

fn context_token(idx: u32) -> u32 {
    if (idx < u.history_count) {
        return history[idx];
    }
    return batch_tokens[u.batch_offset + idx - u.history_count];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.history_count + u.batch_count;
    var start = 0u;
    if (u.repetition_penalty_window > 0u) {
        start = total - min(total, u.repetition_penalty_window);
    }
    if (idx < start || idx >= total) {
        return;
    }
    let token_id = context_token(idx);
    if (token_id >= u.vocab_size) {
        return;
    }
    for (var prior = start; prior < idx; prior++) {
        if (context_token(prior) == token_id) {
            return;
        }
    }
    let logit = f32(logits[token_id]);
    let repeated = select(logit * u.penalty, logit / u.penalty, logit > 0.0);
    logits[token_id] = f32(repeated - u.presence_penalty);
}
