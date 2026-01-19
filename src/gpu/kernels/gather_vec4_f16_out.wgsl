// gather_vec4_f16_out.wgsl

/**
 * Gather Kernel (vec4, F16 output)
 *
 * Vectorized gather for F32 embeddings -> F16 output.
 */

enable f16;

// Tunable workgroup size
override WORKGROUP_SIZE_VEC4: u32 = 64u;

struct Uniforms {
    num_tokens: u32,      // Number of tokens to gather
    hidden_size: u32,     // Embedding dimension
    vocab_size: u32,      // Vocabulary size (for bounds checking)
    transpose: u32,       // 1 if embeddings are [hidden_size, vocab_size] (GGUF layout), 0 otherwise
    index_offset: u32,    // Starting index into indices buffer
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;     // Token IDs [num_tokens]
@group(0) @binding(2) var<storage, read> embeddings: array<f32>;  // Embedding matrix [vocab_size, hidden_size]
@group(0) @binding(4) var<storage, read_write> output_f16: array<f16>;

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
    let token_id = indices[token_idx + u.index_offset];

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
        output_f16[out_base] = f16(embeddings[(dim_base) * u.vocab_size + token_id]);
        output_f16[out_base + 1u] = f16(embeddings[(dim_base + 1u) * u.vocab_size + token_id]);
        output_f16[out_base + 2u] = f16(embeddings[(dim_base + 2u) * u.vocab_size + token_id]);
        output_f16[out_base + 3u] = f16(embeddings[(dim_base + 3u) * u.vocab_size + token_id]);
    } else {
        let embed_base = token_id * u.hidden_size + dim_base;
        output_f16[out_base] = f16(embeddings[embed_base]);
        output_f16[out_base + 1u] = f16(embeddings[embed_base + 1u]);
        output_f16[out_base + 2u] = f16(embeddings[embed_base + 2u]);
        output_f16[out_base + 3u] = f16(embeddings[embed_base + 3u]);
    }
}
