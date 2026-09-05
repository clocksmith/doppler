// T5 attention: unscaled QK, source-owned bucket bias, explicit key mask.
override WORKGROUP_SIZE: u32 = 64u;
override QUERY_LENGTH: u32 = 33u;
override KEY_LENGTH: u32 = 33u;
override NUM_HEADS: u32 = 4u;
override HEAD_DIM: u32 = 64u;
override NUM_BUCKETS: u32 = 32u;
override MAX_DISTANCE: u32 = 128u;
override HAS_RELATIVE_BIAS: bool = true;
override BIDIRECTIONAL: bool = true;
const MAX_KEYS: u32 = 129u;
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> key_mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
fn relative_bucket(q: u32, k: u32) -> u32 {
    let relative = i32(k) - i32(q);
    var buckets = NUM_BUCKETS;
    var base = 0u;
    var distance = u32(max(-relative, 0));
    if (BIDIRECTIONAL) {
        buckets /= 2u;
        if (relative > 0) { base = buckets; }
        distance = u32(abs(relative));
    }
    let exact = buckets / 2u;
    if (distance < exact) { return base + distance; }
    let large = exact + u32(log(f32(distance) / f32(exact)) / log(f32(MAX_DISTANCE) / f32(exact)) * f32(buckets - exact));
    return base + min(large, buckets - 1u);
}
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= QUERY_LENGTH * NUM_HEADS || KEY_LENGTH > MAX_KEYS) { return; }
    let q = index / NUM_HEADS;
    let head = index % NUM_HEADS;
    let width = NUM_HEADS * HEAD_DIM;
    var scores: array<f32, MAX_KEYS>;
    var maximum = -3.402823e38;
    for (var k = 0u; k < KEY_LENGTH; k++) {
        var score = 0.0;
        for (var d = 0u; d < HEAD_DIM; d++) { score += query[q * width + head * HEAD_DIM + d] * key[k * width + head * HEAD_DIM + d]; }
        if (HAS_RELATIVE_BIAS) { score += bias[relative_bucket(q, k) * NUM_HEADS + head]; }
        if (key_mask[k] == 0.0) { score = -3.402823e38; }
        scores[k] = score; maximum = max(maximum, score);
    }
    var total = 0.0;
    for (var k = 0u; k < KEY_LENGTH; k++) {
        scores[k] = select(0.0, exp(scores[k] - maximum), key_mask[k] > 0.0);
        total += scores[k];
    }
    for (var d = 0u; d < HEAD_DIM; d++) {
        var sum = 0.0;
        for (var k = 0u; k < KEY_LENGTH; k++) { sum += scores[k] / total * value[k * width + head * HEAD_DIM + d]; }
        output[q * width + head * HEAD_DIM + d] = sum;
    }
}
