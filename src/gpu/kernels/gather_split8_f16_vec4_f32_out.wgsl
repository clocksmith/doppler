// gather_split8_f16_vec4_f32_out.wgsl

/**
 * Gather Kernel (split8 F16 input -> F32 output, vec4)
 *
 * Row-major embedding lookup for logical embedding tables split across up to
 * eight GPU buffers to stay under maxStorageBufferBindingSize.
 */

enable f16;

override WORKGROUP_SIZE_VEC4: u32 = 64u;

struct Uniforms {
    num_tokens: u32,
    hidden_size: u32,
    vocab_size: u32,
    index_offset: u32,
    input_hidden_size: u32,
    hidden_offset: u32,
    section0_rows: u32,
    section1_rows: u32,
    section2_rows: u32,
    section3_rows: u32,
    section4_rows: u32,
    section5_rows: u32,
    section6_rows: u32,
    section7_rows: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> embeddings0: array<f16>;
@group(0) @binding(3) var<storage, read> embeddings1: array<f16>;
@group(0) @binding(4) var<storage, read> embeddings2: array<f16>;
@group(0) @binding(5) var<storage, read> embeddings3: array<f16>;
@group(0) @binding(6) var<storage, read> embeddings4: array<f16>;
@group(0) @binding(7) var<storage, read> embeddings5: array<f16>;
@group(0) @binding(8) var<storage, read> embeddings6: array<f16>;
@group(0) @binding(9) var<storage, read> embeddings7: array<f16>;
@group(0) @binding(10) var<storage, read_write> output_f32: array<f32>;

fn write_zero(out_base: u32) {
    output_f32[out_base] = 0.0;
    output_f32[out_base + 1u] = 0.0;
    output_f32[out_base + 2u] = 0.0;
    output_f32[out_base + 3u] = 0.0;
}

@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn gather_vec4_f32_out(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        write_zero(out_base);
        return;
    }

    let section1_start = u.section0_rows;
    let section2_start = section1_start + u.section1_rows;
    let section3_start = section2_start + u.section2_rows;
    let section4_start = section3_start + u.section3_rows;
    let section5_start = section4_start + u.section4_rows;
    let section6_start = section5_start + u.section5_rows;
    let section7_start = section6_start + u.section6_rows;
    let section_end = section7_start + u.section7_rows;
    let dim_base = u.hidden_offset + vec4_idx * 4u;

    var local_token: u32;
    var embed_base: u32;
    if (token_id < section1_start) {
        local_token = token_id;
        embed_base = local_token * u.input_hidden_size + dim_base;
        output_f32[out_base] = f32(embeddings0[embed_base]);
        output_f32[out_base + 1u] = f32(embeddings0[embed_base + 1u]);
        output_f32[out_base + 2u] = f32(embeddings0[embed_base + 2u]);
        output_f32[out_base + 3u] = f32(embeddings0[embed_base + 3u]);
    } else if (token_id < section2_start) {
        local_token = token_id - section1_start;
        embed_base = local_token * u.input_hidden_size + dim_base;
        output_f32[out_base] = f32(embeddings1[embed_base]);
        output_f32[out_base + 1u] = f32(embeddings1[embed_base + 1u]);
        output_f32[out_base + 2u] = f32(embeddings1[embed_base + 2u]);
        output_f32[out_base + 3u] = f32(embeddings1[embed_base + 3u]);
    } else if (token_id < section3_start) {
        local_token = token_id - section2_start;
        embed_base = local_token * u.input_hidden_size + dim_base;
        output_f32[out_base] = f32(embeddings2[embed_base]);
        output_f32[out_base + 1u] = f32(embeddings2[embed_base + 1u]);
        output_f32[out_base + 2u] = f32(embeddings2[embed_base + 2u]);
        output_f32[out_base + 3u] = f32(embeddings2[embed_base + 3u]);
    } else if (token_id < section4_start) {
        local_token = token_id - section3_start;
        embed_base = local_token * u.input_hidden_size + dim_base;
        output_f32[out_base] = f32(embeddings3[embed_base]);
        output_f32[out_base + 1u] = f32(embeddings3[embed_base + 1u]);
        output_f32[out_base + 2u] = f32(embeddings3[embed_base + 2u]);
        output_f32[out_base + 3u] = f32(embeddings3[embed_base + 3u]);
    } else if (token_id < section5_start) {
        local_token = token_id - section4_start;
        embed_base = local_token * u.input_hidden_size + dim_base;
        output_f32[out_base] = f32(embeddings4[embed_base]);
        output_f32[out_base + 1u] = f32(embeddings4[embed_base + 1u]);
        output_f32[out_base + 2u] = f32(embeddings4[embed_base + 2u]);
        output_f32[out_base + 3u] = f32(embeddings4[embed_base + 3u]);
    } else if (token_id < section6_start) {
        local_token = token_id - section5_start;
        embed_base = local_token * u.input_hidden_size + dim_base;
        output_f32[out_base] = f32(embeddings5[embed_base]);
        output_f32[out_base + 1u] = f32(embeddings5[embed_base + 1u]);
        output_f32[out_base + 2u] = f32(embeddings5[embed_base + 2u]);
        output_f32[out_base + 3u] = f32(embeddings5[embed_base + 3u]);
    } else if (token_id < section7_start) {
        local_token = token_id - section6_start;
        embed_base = local_token * u.input_hidden_size + dim_base;
        output_f32[out_base] = f32(embeddings6[embed_base]);
        output_f32[out_base + 1u] = f32(embeddings6[embed_base + 1u]);
        output_f32[out_base + 2u] = f32(embeddings6[embed_base + 2u]);
        output_f32[out_base + 3u] = f32(embeddings6[embed_base + 3u]);
    } else if (token_id < section_end) {
        local_token = token_id - section7_start;
        embed_base = local_token * u.input_hidden_size + dim_base;
        output_f32[out_base] = f32(embeddings7[embed_base]);
        output_f32[out_base + 1u] = f32(embeddings7[embed_base + 1u]);
        output_f32[out_base + 2u] = f32(embeddings7[embed_base + 2u]);
        output_f32[out_base + 3u] = f32(embeddings7[embed_base + 3u]);
    } else {
        write_zero(out_base);
    }
}
