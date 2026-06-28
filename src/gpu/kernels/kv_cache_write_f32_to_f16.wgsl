enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    src_offset: u32,
    dst_offset: u32,
    count: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input_k: array<f32>;
@group(0) @binding(2) var<storage, read> input_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_k: array<f16>;
@group(0) @binding(4) var<storage, read_write> output_v: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let i = gid.x + gid.y * num_wg.x * WORKGROUP_SIZE;
    if (i >= u.count) {
        return;
    }

    let src = u.src_offset + i;
    let dst = u.dst_offset + i;
    output_k[dst] = f16(input_k[src]);
    output_v[dst] = f16(input_v[src]);
}
