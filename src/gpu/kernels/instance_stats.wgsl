// Population moments over an explicitly masked numeric context.
override WORKGROUP_SIZE: u32 = 64u;
override CONTEXT_LENGTH: u32 = 512u;
override ZERO_SCALE_EPS: f32 = 0.00001;
@group(0) @binding(0) var<storage, read> context: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> moments: array<f32>;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var count = 0.0;
    var sum = 0.0;
    for (var i = 0u; i < CONTEXT_LENGTH; i++) {
        if (mask[i] > 0.0) { count += 1.0; sum += context[i]; }
    }
    let mean = sum / max(count, 1.0);
    var squared = 0.0;
    for (var i = 0u; i < CONTEXT_LENGTH; i++) {
        if (mask[i] > 0.0) { let d = context[i] - mean; squared += d * d; }
    }
    var scale = sqrt(squared / max(count, 1.0));
    if (count == 0.0) { scale = 1.0; }
    if (scale == 0.0) { scale = ZERO_SCALE_EPS; }
    moments[0] = mean;
    moments[1] = scale;
}
