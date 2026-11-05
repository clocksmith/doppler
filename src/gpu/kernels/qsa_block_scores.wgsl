// qwen4_exp QSA block scores: for query token t with V = start_pos + t + 1
// visible tokens and V / COMPRESS_RATIO complete blocks,
//   score[t, j] = (sum over heads of relu(q[t, h] . block_key[j])) / sqrt(HEAD_DIM)
// with the head sum in head order and the divisor rounded to f32 like the
// source's tensor / math.sqrt(...). Queries at or below the dense visible limit
// select every visible token and write no scores.

override WORKGROUP_SIZE: u32 = 256u;
override QUERY_HEADS: u32;
override HEAD_DIM: u32;
override COMPRESS_RATIO: u32;

const MAX_QUERY_VALUES: u32 = 2048u;

struct Uniforms {
    num_tokens: u32,
    start_pos: u32,
    block_stride: u32,       // row stride of the score scratch (blocks)
    dense_visible_limit: u32,
    score_divisor: f32,      // f32(sqrt(HEAD_DIM))
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> query: array<f32>;      // [num_tokens, QUERY_HEADS, HEAD_DIM]
@group(0) @binding(2) var<storage, read> block_keys: array<f32>; // [blocks, HEAD_DIM]
@group(0) @binding(3) var<storage, read_write> scores: array<f32>; // [num_tokens, block_stride]

var<workgroup> shared_query: array<f32, MAX_QUERY_VALUES>;

fn divide_refined(x: f32, y: f32) -> f32 {
  let estimate = 1.0 / y;
  let inverse = fma(estimate, fma(-y, estimate, 1.0), estimate);
  let quotient = x * inverse;
  return fma(fma(-quotient, y, x), inverse, quotient);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let token = wid.y;
    let values = QUERY_HEADS * HEAD_DIM;
    if (token >= u.num_tokens || values > MAX_QUERY_VALUES) {
        return;
    }
    let visible = u.start_pos + token + 1u;
    if (visible <= u.dense_visible_limit) {
        return;
    }
    let complete_blocks = visible / COMPRESS_RATIO;
    for (var i = lid.x; i < values; i = i + WORKGROUP_SIZE) {
        shared_query[i] = query[token * values + i];
    }
    workgroupBarrier();

    let block = wid.x * WORKGROUP_SIZE + lid.x;
    if (block >= complete_blocks) {
        return;
    }
    var total = 0.0;
    for (var head = 0u; head < QUERY_HEADS; head = head + 1u) {
        var dot = 0.0;
        for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
            dot = fma(shared_query[head * HEAD_DIM + d], block_keys[block * HEAD_DIM + d], dot);
        }
        total = total + max(dot, 0.0);
    }
    scores[token * u.block_stride + block] = divide_refined(total, u.score_divisor);
}
