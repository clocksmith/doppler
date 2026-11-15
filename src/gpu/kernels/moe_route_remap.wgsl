override WORKGROUP_SIZE: u32 = 256u;
struct RemapUniforms {
    num_routes: u32,
    source_experts: u32,
    slot_count: u32,
    _pad0: u32,
}
@group(0) @binding(0) var<uniform> u: RemapUniforms;
@group(0) @binding(1) var<storage, read> source_indices: array<u32>;
@group(0) @binding(2) var<storage, read> source_to_slot: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn remap_routes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.num_routes) { return; }
    let source = source_indices[gid.x];
    var slot = 0xffffffffu;
    if (source < u.source_experts) {
        let candidate = source_to_slot[source];
        if (candidate < u.slot_count) { slot = candidate; }
    }
    output[gid.x] = slot;
}
