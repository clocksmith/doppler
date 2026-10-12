// Exact representation expansion; source bits are not changed.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms { size: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
    let i = gid.x + gid.y * num_wg.x * WORKGROUP_SIZE;
    if (i >= u.size) { return; }
    output[i] = vec2<u32>(input[i], 0u);
}
