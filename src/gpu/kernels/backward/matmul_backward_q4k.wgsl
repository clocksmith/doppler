// Frozen Q4_K row-wise weight input gradient.
//
// Forward:  Y[M,N] = X[M,K] @ W[N,K]^T
// Backward: dX[M,K] = dY[M,N] @ W[N,K]

// The packed base weight remains read-only. Each invocation owns one dX cell
// and dequantizes only the matching K element from every output row.

const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;

override TILE_SIZE: u32 = 16u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transpose_b: u32,
    num_blocks_per_row: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Q4KBlock {
    d_dmin: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read_write> dX: array<f32>;

fn get_scale_byte(scales: array<u32, 3>, byte_index: u32) -> u32 {
    let word_index = byte_index / 4u;
    let byte_in_word = byte_index % 4u;
    return (scales[word_index] >> (byte_in_word * 8u)) & 0xffu;
}

fn get_scale_min(scales: array<u32, 3>, subblock: u32) -> vec2<u32> {
    var scale: u32;
    var minimum: u32;
    if (subblock < 4u) {
        scale = get_scale_byte(scales, subblock) & 63u;
        minimum = get_scale_byte(scales, subblock + 4u) & 63u;
    } else {
        let packed = get_scale_byte(scales, subblock + 4u);
        scale = (packed & 0x0fu) | ((get_scale_byte(scales, subblock - 4u) >> 6u) << 4u);
        minimum = (packed >> 4u) | ((get_scale_byte(scales, subblock) >> 6u) << 4u);
    }
    return vec2<u32>(scale, minimum);
}

fn get_q4(qs: array<u32, 32>, element: u32) -> u32 {
    let chunk = element / 64u;
    let chunk_element = element % 64u;
    let upper = chunk_element >= 32u;
    let byte_in_chunk = select(chunk_element, chunk_element - 32u, upper);
    let byte_index = chunk * 32u + byte_in_chunk;
    let byte_value = (qs[byte_index / 4u] >> ((byte_index % 4u) * 8u)) & 0xffu;
    return select(byte_value & 0x0fu, (byte_value >> 4u) & 0x0fu, upper);
}

fn dequantize_weight(row: u32, column: u32) -> f32 {
    let block_in_row = column / QK_K;
    let element = column % QK_K;
    let block = W[row * u.num_blocks_per_row + block_in_row];
    let d = unpack2x16float(block.d_dmin).x;
    let dmin = unpack2x16float(block.d_dmin).y;
    let scale_min = get_scale_min(block.scales, element / SUBBLOCK_SIZE);
    return d * f32(scale_min.x) * f32(get_q4(block.qs, element))
        - dmin * f32(scale_min.y);
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let column = global_id.y;
    if (row >= u.M || column >= u.K || u.transpose_b != 1u) {
        return;
    }

    var sum = 0.0f;
    for (var output_column = 0u; output_column < u.N; output_column += 1u) {
        sum += dY[row * u.N + output_column] * dequantize_weight(output_column, column);
    }
    dX[row * u.K + column] = sum * u.alpha;
}
