// AUTO-GENERATED from src/gpu/kernels/bias_add.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// bias_add_f16.wgsl

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
    token_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

override WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let token = gid.z * max(u.token_stride, 1u) + gid.y;
    if (token >= u.num_tokens || d >= u.dim) {
        return;
    }

    // Convert byte offsets to F16 indices
    let data_base = u.data_offset / 2u;
    let bias_base = u.bias_offset / 2u;
    let idx = token * u.dim + d;
    let out = f32(data[data_base + idx]) + f32(bias[bias_base + d]);
    data[data_base + idx] = f16(out);
}
