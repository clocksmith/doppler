/**
 * Bias Add Kernel
 *
 * Adds per-channel bias to a 2D tensor in-place.
 *
 * data layout: [numTokens, dim] flattened, with optional byte offset
 * bias layout: [N, dim] where we select slice at bias_offset
 */

struct Uniforms {
    num_tokens: u32,
    dim: u32,
    data_offset: u32,  // byte offset into data buffer (divide by 4 for F32)
    bias_offset: u32,  // byte offset into bias buffer (divide by 4 for F32)
}

override WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_tokens * u.dim;
    if (idx >= total) {
        return;
    }

    // Convert byte offsets to F32 indices
    let data_base = u.data_offset / 4u;
    let bias_base = u.bias_offset / 4u;

    let d = idx % u.dim;
    data[data_base + idx] = data[data_base + idx] + bias[bias_base + d];
}

