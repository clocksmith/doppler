// gather_vec4.wgsl

/**
 * Gather Kernel (vec4) - Token Embedding Lookup
 *
 * Vectorized gather for F32 embeddings -> F32 output.
 */

// Tunable workgroup size
override WORKGROUP_SIZE_VEC4: u32 = 64u;

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
@group(0) @binding(1) var<storage, read> indices: array<u32>;      // Token IDs [num_tokens]
@group(0) @binding(2) var<storage, read> embeddings: array<f32>;   // Embedding matrix [vocab_size, hidden_size]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // Output [num_tokens, hidden_size]

// Vectorized version for better memory throughput
@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn gather_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let vec4_per_row = u.hidden_size / 4u;
    let total_vec4s = u.num_tokens * vec4_per_row;

    if (tid >= total_vec4s) {
        return;
    }

    // Compute token index and vec4 index within row
    let token_idx = tid / vec4_per_row;
    let vec4_idx = tid % vec4_per_row;

    // Get the token ID
    let token_id = indices[token_idx + u.index_offset];

    // Bounds check
    if (token_id >= u.vocab_size) {
        let out_base = tid * 4u;
        output[out_base] = 0.0;
        output[out_base + 1u] = 0.0;
        output[out_base + 2u] = 0.0;
        output[out_base + 3u] = 0.0;
        return;
    }

    // Gather 4 elements
    let out_base = tid * 4u;
    let dim_base = vec4_idx * 4u;

    if (u.transpose == 1u) {
        // Transposed layout [hidden_size, vocab_size]: elements are strided by vocab_size
        output[out_base] = embeddings[(dim_base) * u.vocab_size + token_id];
        output[out_base + 1u] = embeddings[(dim_base + 1u) * u.vocab_size + token_id];
        output[out_base + 2u] = embeddings[(dim_base + 2u) * u.vocab_size + token_id];
        output[out_base + 3u] = embeddings[(dim_base + 3u) * u.vocab_size + token_id];
    } else {
        // Standard layout [vocab_size, hidden_size]: elements are contiguous
        let embed_base = token_id * u.hidden_size + dim_base;
        output[out_base] = embeddings[embed_base];
        output[out_base + 1u] = embeddings[embed_base + 1u];
        output[out_base + 2u] = embeddings[embed_base + 2u];
        output[out_base + 3u] = embeddings[embed_base + 3u];
    }
}
