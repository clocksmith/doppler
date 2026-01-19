// cross_entropy_loss.wgsl

/**
 * Cross-entropy loss kernel (expects softmax input).
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    vocab_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> softmax: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const EPSILON: f32 = 1e-9;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= u.num_tokens) {
        return;
    }

    let target = targets[token_idx];
    if (target >= u.vocab_size) {
        output[token_idx] = 0.0;
        return;
    }

    let offset = token_idx * u.vocab_size + target;
    let p = max(softmax[offset], EPSILON);
    output[token_idx] = -log(p);
}
