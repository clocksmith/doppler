// gather_f16_out.wgsl

/**
 * Gather Kernel (F16 output)
 *
 * F32 embeddings -> F16 output for reduced memory bandwidth.
 */

enable f16;

// Tunable workgroup size
override WORKGROUP_SIZE_MAIN: u32 = 256u;

struct Uniforms {
    num_tokens: u32,      // Number of tokens to gather
    hidden_size: u32,     // Embedding dimension
    vocab_size: u32,      // Vocabulary size (for bounds checking)
    transpose: u32,       // 1 if embeddings are [hidden_size, vocab_size] (GGUF layout), 0 otherwise
    index_offset: u32,    // Starting index into indices buffer
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;     // Token IDs [num_tokens]
@group(0) @binding(2) var<storage, read> embeddings: array<f32>;  // Embedding matrix [vocab_size, hidden_size]
@group(0) @binding(4) var<storage, read_write> output_f16: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn gather_f16_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_elements = u.num_tokens * u.hidden_size;

    if (tid >= total_elements) {
        return;
    }

    let token_idx = tid / u.hidden_size;
    let dim_idx = tid % u.hidden_size;
    let token_id = indices[token_idx + u.index_offset];

    if (token_id >= u.vocab_size) {
        output_f16[tid] = f16(0.0);
        return;
    }

    var embed_offset: u32;
    if (u.transpose == 1u) {
        embed_offset = dim_idx * u.vocab_size + token_id;
    } else {
        embed_offset = token_id * u.hidden_size + dim_idx;
    }
    output_f16[tid] = f16(embeddings[embed_offset]);
}
