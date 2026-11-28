// Full-parameter AdamW with one shared gradient-norm clip.
override COUNT: u32 = 1u;
override PARTIAL_OFFSET: u32 = 0u;
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read_write> first: array<f32>;
@group(0) @binding(3) var<storage, read_write> second: array<f32>;
@group(0) @binding(4) var<storage, read_write> clip_scale: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
struct Step {
    iteration: f32,
    rate: f32,
    _pad0: vec2<f32>,
}
@group(0) @binding(6) var<uniform> step: Step;
var<workgroup> values: array<f32, 256>;

fn sum_group(value: f32, lane: u32) -> f32 {
    values[lane] = value;
    workgroupBarrier();
    for (var shift = 128u; shift > 0u; shift >>= 1u) {
        if (lane < shift) { values[lane] += values[lane + shift]; }
        workgroupBarrier();
    }
    return values[0];
}

@compute @workgroup_size(256)
fn norm_part(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
             @builtin(local_invocation_index) lane: u32) {
    let chunk = group.y * grid.x + group.x;
    let start = chunk * 1024u;
    if (start >= COUNT) { return; }
    var sum = 0.0;
    for (var i = start + lane; i < min(start + 1024u, COUNT); i += 256u) {
        sum += gradient[i] * gradient[i];
    }
    let subtotal = sum_group(sum, lane);
    if (lane == 0u) { partials[PARTIAL_OFFSET + chunk] = subtotal; }
}

@compute @workgroup_size(256)
fn norm_final(@builtin(local_invocation_index) lane: u32) {
    var sum = 0.0;
    for (var i = lane; i < COUNT; i += 256u) { sum += partials[i]; }
    let norm = sqrt(sum_group(sum, lane));
    if (lane == 0u) { clip_scale[0] = 1.0 / max(1.0, norm); }
}

@compute @workgroup_size(256)
fn adam(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
    let i = id.y * grid.x * 256u + id.x;
    if (i >= COUNT) { return; }
    let g = gradient[i] * clip_scale[0];
    let m = 0.9 * first[i] + 0.1 * g;
    let v = 0.99 * second[i] + 0.01 * g * g;
    first[i] = m;
    second[i] = v;
    let m_hat = m / (1.0 - pow(0.9, step.iteration));
    let v_hat = v / (1.0 - pow(0.99, step.iteration));
    let p = params[i];
    params[i] = p - step.rate * (m_hat / (sqrt(v_hat) + 1e-8) + 0.01 * p);
}
