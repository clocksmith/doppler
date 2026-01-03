// embed_backward.wgsl

/**
 * Embedding backward kernel.
 *
 * Accumulates dL/d(embed_table) from dL/d(embed_out) and token indices.
 *
 * Forward gather (non-transposed weights):
 *   out[token, dim] = embed_table[token_id, dim]
 *
 * Backward:
 *   grad_embed[token_id, dim] += grad_out[token, dim]
 *
 * Note: Multiple tokens can reference the same token_id, so accumulation
 * uses atomic add via atomic<u32> compare-exchange on f32 bit patterns.
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    hidden_size: u32,
    vocab_size: u32,
    transpose: u32,
    index_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_embeddings: array<atomic<u32>>;

fn atomic_add_f32(ptr: ptr<storage, atomic<u32>, read_write>, delta: f32) {
    var old_bits = atomicLoad(ptr);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_val + delta);
        let res = atomicCompareExchangeWeak(ptr, old_bits, new_bits);
        if (res.exchanged) {
            break;
        }
        old_bits = res.old_value;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_tokens * u.hidden_size;
    if (idx >= total) {
        return;
    }

    let token_idx = idx / u.hidden_size;
    let dim_idx = idx % u.hidden_size;

    let token_id = indices[token_idx + u.index_offset];
    if (token_id >= u.vocab_size) {
        return;
    }

    var out_idx: u32;
    if (u.transpose == 1u) {
        out_idx = dim_idx * u.vocab_size + token_id;
    } else {
        out_idx = token_id * u.hidden_size + dim_idx;
    }

    atomic_add_f32(&grad_embeddings[out_idx], grad_output[idx]);
}
