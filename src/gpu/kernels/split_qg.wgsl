// split_qg.wgsl

/**
 * De-interleave Q and Gate projections from q_proj output for attentionOutputGate models.
 *
 * Models like Qwen 3.5 store q_proj weights with interleaved head layout:
 *   rows [h*headDim*2 : h*headDim*2+headDim]     = Q for head h
 *   rows [h*headDim*2+headDim : (h+1)*headDim*2] = Gate for head h
 *
 * A single full matmul over all 2*qSize rows produces interleaved output:
 *   input[token, h*headDim*2 : h*headDim*2+headDim]     = Q head h
 *   input[token, h*headDim*2+headDim : (h+1)*headDim*2] = Gate head h
 *
 * This kernel separates them into contiguous Q and G outputs:
 *   Q[token, h*headDim + dim] = input[token, h*headDim*2 + dim]
 *   G[token, h*headDim + dim] = input[token, h*headDim*2 + headDim + dim]
 *
 * Input layout  (row-major): [numTokens, numHeads * headDim * 2]
 * Output Q layout (row-major): [numTokens, numHeads * headDim]
 * Output G layout (row-major): [numTokens, numHeads * headDim]
 */

struct Params {
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    _pad: u32,
}

override WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> Q: array<f32>;
@group(0) @binding(3) var<storage, read_write> G: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let q_size = params.num_heads * params.head_dim;
    let total_elements = params.num_tokens * q_size;

    if (idx >= total_elements) {
        return;
    }

    let token = idx / q_size;
    let elem = idx % q_size;
    let head = elem / params.head_dim;
    let dim = elem % params.head_dim;

    // Input is interleaved per head: [Q_h (headDim elems), G_h (headDim elems)]
    let src_q = token * (q_size * 2u) + head * (params.head_dim * 2u) + dim;
    let src_g = src_q + params.head_dim;

    Q[idx] = input[src_q];
    G[idx] = input[src_g];
}
