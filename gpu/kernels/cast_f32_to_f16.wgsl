// Cast Kernel - f32 to f16
//
// Converts a flat f32 buffer to f16.
// Used for KV-cache compression and future f16 pipelines.
// Supports 2D dispatch for large tensors (>65535 workgroups).

enable f16;

const WG_SIZE: u32 = 256u;
const MAX_WG_X: u32 = 65535u;

struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    // Support 2D dispatch for large tensors
    // Global index = x + y * (numWorkgroupsX * workgroupSize)
    let i = gid.x + gid.y * num_wg.x * WG_SIZE;
    if (i >= uniforms.size) {
        return;
    }
    output[i] = f16(input[i]);
}

