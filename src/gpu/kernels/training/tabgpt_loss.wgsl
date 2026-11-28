// Per-token log-sum-exp loss and gradient of the batch mean.
override VOCAB: u32 = 1u;
override ROWS: u32 = 1u;
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<storage, read_write> dlogits: array<f32>;
var<workgroup> values: array<f32, 128>;

@compute @workgroup_size(128)
fn loss(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let base = group.x * VOCAB;
    var largest = -3.4e38f;
    for (var v = lane; v < VOCAB; v += 128u) { largest = max(largest, logits[base + v]); }
    values[lane] = largest;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { values[lane] = max(values[lane], values[lane + stride]); }
        workgroupBarrier();
    }
    let pivot = values[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var v = lane; v < VOCAB; v += 128u) {
        let p = exp(logits[base + v] - pivot);
        dlogits[base + v] = p;
        sum += p;
    }
    values[lane] = sum;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { values[lane] += values[lane + stride]; }
        workgroupBarrier();
    }
    let normalizer = values[0];
    let target_id = targets[group.x];
    if (lane == 0u) { losses[group.x] = (pivot - logits[base + target_id]) + log(normalizer); }
    for (var v = lane; v < VOCAB; v += 128u) {
        let p = dlogits[base + v] / normalizer;
        dlogits[base + v] = (p - select(0.0, 1.0, v == target_id)) / f32(ROWS);
    }
}
