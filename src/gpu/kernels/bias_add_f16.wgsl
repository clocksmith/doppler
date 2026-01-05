/**
 * Bias Add Kernel (F16)
 *
 * Adds per-channel bias to a 2D tensor in-place.
 *
 * data layout: [numTokens, dim] flattened, with optional byte offset
 * bias layout: [N, dim] where we select slice at bias_offset
 */

enable f16;

struct Uniforms {
    num_tokens: u32,
    dim: u32,
    data_offset: u32,  // byte offset into data buffer (divide by 2 for F16)
    bias_offset: u32,  // byte offset into bias buffer (divide by 2 for F16)
}

override WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_tokens * u.dim;
    if (idx >= total) {
        return;
    }

    // Convert byte offsets to F16 indices
    let data_base = u.data_offset / 2u;
    let bias_base = u.bias_offset / 2u;

    let d = idx % u.dim;
    let out = f32(data[data_base + idx]) + f32(bias[bias_base + d]);
    data[data_base + idx] = f16(out);
}
