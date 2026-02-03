// gather_f16.wgsl

/**
 * Gather Kernel (F16) - Token Embedding Lookup with F16 Embeddings
 *
 * Gathers rows from an F16 embedding matrix based on token indices.
 * Outputs F32 for downstream computation (activations are F32).
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
@group(0) @binding(1) var<storage, read> indices: array<u32>;      // Token IDs [num_tokens]
@group(0) @binding(2) var<storage, read> embeddings: array<f16>;   // F16 Embedding matrix [vocab_size, hidden_size]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // F32 Output [num_tokens, hidden_size]

@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_elements = u.num_tokens * u.hidden_size;

    if (tid >= total_elements) {
        return;
    }

    // Compute token index and dimension index
    let token_idx = tid / u.hidden_size;
    let dim_idx = tid % u.hidden_size;

    // Get the token ID (with bounds check)
    let token_id = indices[token_idx + u.index_offset];

    // Bounds check on vocab
    if (token_id >= u.vocab_size) {
        output[tid] = 0.0;
        return;
    }

    // Gather from F16 embedding matrix, convert to F32 output
    // For GGUF layout [hidden_size, vocab_size]: offset = dim_idx * vocab_size + token_id
    // For standard layout [vocab_size, hidden_size]: offset = token_id * hidden_size + dim_idx
    var embed_offset: u32;
    if (u.transpose == 1u) {
        embed_offset = dim_idx * u.vocab_size + token_id;
    } else {
        embed_offset = token_id * u.hidden_size + dim_idx;
    }
    output[tid] = f32(embeddings[embed_offset]);
}
