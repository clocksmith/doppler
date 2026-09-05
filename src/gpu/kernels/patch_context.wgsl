// Concatenate standardized values and observation masks per numeric patch.
override WORKGROUP_SIZE: u32 = 64u;
override CONTEXT_LENGTH: u32 = 512u;
override PATCH_SIZE: u32 = 16u;
override PATCH_COUNT: u32 = 32u;
@group(0) @binding(0) var<storage, read> context: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> moments: array<f32>;
@group(0) @binding(3) var<storage, read_write> patches: array<f32>;
@group(0) @binding(4) var<storage, read_write> attention_mask: array<f32>;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= PATCH_COUNT * PATCH_SIZE * 2u) { return; }
    let patch_index = i / (PATCH_SIZE * 2u);
    let column = i % (PATCH_SIZE * 2u);
    let source = patch_index * PATCH_SIZE + column % PATCH_SIZE;
    var value = 0.0;
    if (source < CONTEXT_LENGTH && mask[source] > 0.0) {
        value = select((context[source] - moments[0]) / moments[1], 1.0, column >= PATCH_SIZE);
    }
    patches[i] = value;
    if (column == 0u) {
        var observed = false;
        for (var d = 0u; d < PATCH_SIZE; d++) { observed = observed || mask[patch_index * PATCH_SIZE + d] > 0.0; }
        attention_mask[patch_index] = select(0.0, 1.0, observed);
    }
    if (i == 0u) { attention_mask[PATCH_COUNT] = 1.0; }
}
