enable f16;

override WORKGROUP_SIZE_VEC4: u32 = 64u;
override WEIGHT_SCALE: f32 = 0.0625;
override STORAGE_OFFSET_BINARY: u32 = 0u;

struct Uniforms {
    num_tokens: u32,
    hidden_size: u32,
    vocab_size: u32,
    transpose: u32,
    index_offset: u32,
    input_hidden_size: u32,
    hidden_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> embeddings_words: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_f16: array<f16>;

fn packed_byte(byte_index: u32) -> u32 {
    let word = embeddings_words[byte_index >> 2u];
    let shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn signed_int4(byte_value: u32, high_nibble: bool) -> f32 {
    let value = select(byte_value & 0x0fu, (byte_value >> 4u) & 0x0fu, high_nibble);
    if (STORAGE_OFFSET_BINARY == 1u) {
        return f32(value) - 8.0;
    }
    return select(f32(value), f32(i32(value) - 16), value >= 8u);
}

fn read_weight(row_base: u32, dim: u32) -> f16 {
    let byte_value = packed_byte(row_base + (dim >> 1u));
    let value = signed_int4(byte_value, (dim & 1u) == 1u) * WEIGHT_SCALE;
    return f16(value);
}

@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn gather_litert_int4_vec4_f16_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let vec4_per_row = u.hidden_size / 4u;
    let total_vec4s = u.num_tokens * vec4_per_row;

    if (tid >= total_vec4s) {
        return;
    }

    let token_idx = tid / vec4_per_row;
    let vec4_idx = tid % vec4_per_row;
    let token_id = indices[token_idx + u.index_offset];
    let out_base = tid * 4u;

    if (token_id >= u.vocab_size) {
        output_f16[out_base] = f16(0.0);
        output_f16[out_base + 1u] = f16(0.0);
        output_f16[out_base + 2u] = f16(0.0);
        output_f16[out_base + 3u] = f16(0.0);
        return;
    }

    let packed_row_bytes = (u.input_hidden_size + 1u) >> 1u;
    let row_base = token_id * packed_row_bytes;
    let dim_base = u.hidden_offset + vec4_idx * 4u;

    output_f16[out_base] = read_weight(row_base, dim_base);
    output_f16[out_base + 1u] = read_weight(row_base, dim_base + 1u);
    output_f16[out_base + 2u] = read_weight(row_base, dim_base + 2u);
    output_f16[out_base + 3u] = read_weight(row_base, dim_base + 3u);
}
