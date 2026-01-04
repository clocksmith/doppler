/**
 * Gather Kernel (F16) - Token Embedding Lookup with F16 Embeddings
 *
 * Gathers rows from an F16 embedding matrix based on token indices.
 * Outputs F32 for downstream computation (activations are F32).
 */

enable f16;

// Tunable workgroup sizes
override WORKGROUP_SIZE_MAIN: u32 = 256u;
override WORKGROUP_SIZE_VEC4: u32 = 64u;

struct Uniforms {
    num_tokens: u32,      // Number of tokens to gather
    hidden_size: u32,     // Embedding dimension
    vocab_size: u32,      // Vocabulary size (for bounds checking)
    transpose: u32,       // 1 if embeddings are [hidden_size, vocab_size] (GGUF layout), 0 otherwise
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
    let token_id = indices[token_idx];

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
    let token_id = indices[token_idx];

    // Bounds check
    if (token_id >= u.vocab_size) {
        let out_base = tid * 4u;
        output[out_base] = 0.0;
        output[out_base + 1u] = 0.0;
        output[out_base + 2u] = 0.0;
        output[out_base + 3u] = 0.0;
        return;
    }

    // Gather 4 F16 elements and convert to F32
    let out_base = tid * 4u;
    let dim_base = vec4_idx * 4u;

    if (u.transpose == 1u) {
        // Transposed layout [hidden_size, vocab_size]: elements are strided by vocab_size
        output[out_base] = f32(embeddings[(dim_base) * u.vocab_size + token_id]);
        output[out_base + 1u] = f32(embeddings[(dim_base + 1u) * u.vocab_size + token_id]);
        output[out_base + 2u] = f32(embeddings[(dim_base + 2u) * u.vocab_size + token_id]);
        output[out_base + 3u] = f32(embeddings[(dim_base + 3u) * u.vocab_size + token_id]);
    } else {
        // Standard layout [vocab_size, hidden_size]: elements are contiguous
        let embed_base = token_id * u.hidden_size + dim_base;
        output[out_base] = f32(embeddings[embed_base]);
        output[out_base + 1u] = f32(embeddings[embed_base + 1u]);
        output[out_base + 2u] = f32(embeddings[embed_base + 2u]);
        output[out_base + 3u] = f32(embeddings[embed_base + 3u]);
    }
}

// =============================================================================
// F16 Input → F16 Output Variants
// For F16 activation mode with F16 embeddings (no dtype conversion)
// =============================================================================

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
    let token_id = indices[token_idx];

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
    // F16 → F16: no conversion needed
    output_f16[tid] = embeddings[embed_offset];
}

@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn gather_vec4_f16_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let vec4_per_row = u.hidden_size / 4u;
    let total_vec4s = u.num_tokens * vec4_per_row;

    if (tid >= total_vec4s) {
        return;
    }

    let token_idx = tid / vec4_per_row;
    let vec4_idx = tid % vec4_per_row;
    let token_id = indices[token_idx];

    if (token_id >= u.vocab_size) {
        let out_base = tid * 4u;
        output_f16[out_base] = f16(0.0);
        output_f16[out_base + 1u] = f16(0.0);
        output_f16[out_base + 2u] = f16(0.0);
        output_f16[out_base + 3u] = f16(0.0);
        return;
    }

    let out_base = tid * 4u;
    let dim_base = vec4_idx * 4u;

    if (u.transpose == 1u) {
        // Transposed layout [hidden_size, vocab_size]: elements are strided
        output_f16[out_base] = embeddings[(dim_base) * u.vocab_size + token_id];
        output_f16[out_base + 1u] = embeddings[(dim_base + 1u) * u.vocab_size + token_id];
        output_f16[out_base + 2u] = embeddings[(dim_base + 2u) * u.vocab_size + token_id];
        output_f16[out_base + 3u] = embeddings[(dim_base + 3u) * u.vocab_size + token_id];
    } else {
        // Standard layout [vocab_size, hidden_size]: elements are contiguous
        let embed_base = token_id * u.hidden_size + dim_base;
        output_f16[out_base] = embeddings[embed_base];
        output_f16[out_base + 1u] = embeddings[embed_base + 1u];
        output_f16[out_base + 2u] = embeddings[embed_base + 2u];
        output_f16[out_base + 3u] = embeddings[embed_base + 3u];
    }
}
