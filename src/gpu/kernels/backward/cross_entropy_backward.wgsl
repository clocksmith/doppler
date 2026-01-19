// cross_entropy_backward.wgsl

/**
 * Cross-entropy backward kernel (expects softmax input).
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
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_tokens * u.vocab_size;
    if (idx >= total) {
        return;
    }

    let token_idx = idx / u.vocab_size;
    let class_idx = idx % u.vocab_size;
    let target = targets[token_idx];
    let grad = softmax[idx];
    if (class_idx == target) {
        grad = grad - 1.0;
    }
    let scale = grad_output[token_idx];
    output[idx] = grad * scale;
}
