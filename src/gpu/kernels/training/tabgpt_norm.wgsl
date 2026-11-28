// Weighted row-wise RMSNorm, its input vector-Jacobian product and weight gradient.
override WIDTH: u32 = 1u;
override ROWS: u32 = 1u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> inv_rms: array<f32>;
@group(0) @binding(4) var<storage, read> dy: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
@group(0) @binding(6) var<storage, read_write> dw: array<f32>;
var<workgroup> sums: array<f32, 128>;

fn row_sum(value: f32, lane: u32) -> f32 {
    sums[lane] = value;
    workgroupBarrier();
    for (var shift = 64u; shift > 0u; shift >>= 1u) {
        if (lane < shift) { sums[lane] += sums[lane + shift]; }
        workgroupBarrier();
    }
    return sums[0];
}

@compute @workgroup_size(128)
fn forward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let start = group.x * WIDTH;
    var squares = 0.0;
    for (var c = lane; c < WIDTH; c += 128u) { squares += x[start + c] * x[start + c]; }
    let scale = inverseSqrt(row_sum(squares, lane) / f32(WIDTH) + 1e-5);
    if (lane == 0u) { inv_rms[group.x] = scale; }
    for (var c = lane; c < WIDTH; c += 128u) { y[start + c] = x[start + c] * scale * weight[c]; }
}

@compute @workgroup_size(128)
fn input_grad(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let start = group.x * WIDTH;
    let scale = inv_rms[group.x];
    var inner = 0.0;
    for (var c = lane; c < WIDTH; c += 128u) {
        inner += dy[start + c] * weight[c] * (x[start + c] * scale);
    }
    let mean = row_sum(inner, lane) / f32(WIDTH);
    for (var c = lane; c < WIDTH; c += 128u) {
        dx[start + c] = (dy[start + c] * weight[c] - x[start + c] * scale * mean) * scale;
    }
}

@compute @workgroup_size(128)
fn weight_grad(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= WIDTH) { return; }
    var sum = 0.0;
    for (var r = 0u; r < ROWS; r++) {
        let i = r * WIDTH + id.x;
        sum += dy[i] * (x[i] * inv_rms[r]);
    }
    dw[id.x] = sum;
}
