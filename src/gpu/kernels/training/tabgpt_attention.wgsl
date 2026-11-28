// Fused Q/K projection layout [batch, time, (Q,K,V), head, channel].
// Probability and score-gradient layout [batch, head, query time, key time].
override BATCH: u32 = 1u;
override LENGTH: u32 = 1u;
override HEADS: u32 = 1u;
override HEAD_WIDTH: u32 = 2u;
@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> inv_rms: array<f32>;
@group(0) @binding(2) var<storage, read_write> dqkv: array<f32>;
@group(0) @binding(3) var<storage, read_write> prob: array<f32>;
@group(0) @binding(4) var<storage, read_write> context_out: array<f32>;
@group(0) @binding(5) var<storage, read> dout: array<f32>;
@group(0) @binding(6) var<storage, read_write> dscores: array<f32>;
var<workgroup> scratch: array<f32, 128>;

fn packed_offset(row: u32, part: u32, head: u32) -> u32 {
    return ((row * 3u + part) * HEADS + head) * HEAD_WIDTH;
}

fn rotate(pair: vec2<f32>, time: u32, channel: u32, sign: f32) -> vec2<f32> {
    let angle = sign * f32(time) * pow(10000.0, -f32(channel) / f32(HEAD_WIDTH));
    let cs = cos(angle);
    let sn = sin(angle);
    return vec2<f32>(pair.x * cs - pair.y * sn, pair.x * sn + pair.y * cs);
}

@compute @workgroup_size(64)
fn position(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x / (2u * HEADS);
    if (row >= BATCH * LENGTH) { return; }
    let base = packed_offset(row, (id.x / HEADS) % 2u, id.x % HEADS);
    var squares = 0.0;
    for (var c = 0u; c < HEAD_WIDTH; c += 2u) {
        let v = rotate(vec2<f32>(qkv[base + c], qkv[base + c + 1u]), row % LENGTH, c, 1.0);
        qkv[base + c] = v.x;
        qkv[base + c + 1u] = v.y;
        squares += dot(v, v);
    }
    let scale = inverseSqrt(squares / f32(HEAD_WIDTH) + 1e-6);
    inv_rms[id.x] = scale;
    for (var c = 0u; c < HEAD_WIDTH; c++) { qkv[base + c] *= scale; }
}

@compute @workgroup_size(64)
fn position_grad(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x / (2u * HEADS);
    if (row >= BATCH * LENGTH) { return; }
    let base = packed_offset(row, (id.x / HEADS) % 2u, id.x % HEADS);
    var inner = 0.0;
    for (var c = 0u; c < HEAD_WIDTH; c++) { inner += qkv[base + c] * dqkv[base + c]; }
    let mean = inner / f32(HEAD_WIDTH);
    let scale = inv_rms[id.x];
    for (var c = 0u; c < HEAD_WIDTH; c += 2u) {
        let i = base + c;
        let dy = vec2<f32>(dqkv[i], dqkv[i + 1u]);
        let y = vec2<f32>(qkv[i], qkv[i + 1u]);
        let g = rotate((dy - y * mean) * scale, row % LENGTH, c, -1.0);
        dqkv[i] = g.x;
        dqkv[i + 1u] = g.y;
    }
}

@compute @workgroup_size(128)
fn scores(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let t = group.x % LENGTH;
    let head = (group.x / LENGTH) % HEADS;
    let batch_start = (group.x / (LENGTH * HEADS)) * LENGTH;
    let query = packed_offset(batch_start + t, 0u, head);
    let row = group.x * LENGTH;
    var largest = -3.4e38f;
    for (var s = lane; s < LENGTH; s += 128u) {
        var value = 0.0;
        if (s <= t) {
            let key = packed_offset(batch_start + s, 1u, head);
            for (var c = 0u; c < HEAD_WIDTH; c++) { value += qkv[query + c] * qkv[key + c]; }
            value *= inverseSqrt(f32(HEAD_WIDTH));
            largest = max(largest, value);
        }
        prob[row + s] = value;
    }
    scratch[lane] = largest;
    workgroupBarrier();
    for (var shift = 64u; shift > 0u; shift >>= 1u) {
        if (lane < shift) { scratch[lane] = max(scratch[lane], scratch[lane + shift]); }
        workgroupBarrier();
    }
    let row_max = scratch[0];
    workgroupBarrier();
    var total = 0.0;
    for (var s = lane; s <= t; s += 128u) {
        let w = exp(prob[row + s] - row_max);
        prob[row + s] = w;
        total += w;
    }
    scratch[lane] = total;
    workgroupBarrier();
    for (var shift = 64u; shift > 0u; shift >>= 1u) {
        if (lane < shift) { scratch[lane] += scratch[lane + shift]; }
        workgroupBarrier();
    }
    for (var s = lane; s <= t; s += 128u) { prob[row + s] /= scratch[0]; }
}

@compute @workgroup_size(256)
fn context(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x / (HEADS * HEAD_WIDTH);
    if (row >= BATCH * LENGTH) { return; }
    let t = row % LENGTH;
    let batch = row / LENGTH;
    let h = (id.x / HEAD_WIDTH) % HEADS;
    let c = id.x % HEAD_WIDTH;
    let att = ((batch * HEADS + h) * LENGTH + t) * LENGTH;
    var total = 0.0;
    for (var s = 0u; s <= t; s++) {
        total += prob[att + s] * qkv[packed_offset(batch * LENGTH + s, 2u, h) + c];
    }
    context_out[id.x] = total;
}

// For O = sum_s p_s V_s, d(q dot k)_s = p_s * (dO dot (V_s - O)) / sqrt(d).
@compute @workgroup_size(256)
fn scores_grad(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x % LENGTH;
    let row = id.x / LENGTH;
    if (row >= BATCH * HEADS * LENGTH) { return; }
    let t = row % LENGTH;
    if (s > t) { dscores[id.x] = 0.0; return; }
    let h = (row / LENGTH) % HEADS;
    let batch_start = (row / (HEADS * LENGTH)) * LENGTH;
    let v = packed_offset(batch_start + s, 2u, h);
    let out = ((batch_start + t) * HEADS + h) * HEAD_WIDTH;
    var dot_product = 0.0;
    for (var c = 0u; c < HEAD_WIDTH; c++) {
        dot_product += dout[out + c] * (qkv[v + c] - context_out[out + c]);
    }
    dscores[id.x] = prob[id.x] * dot_product * inverseSqrt(f32(HEAD_WIDTH));
}

@compute @workgroup_size(256)
fn qkv_grad(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x / (HEADS * HEAD_WIDTH);
    if (row >= BATCH * LENGTH) { return; }
    let t = row % LENGTH;
    let b = row / LENGTH;
    let h = (id.x / HEAD_WIDTH) % HEADS;
    let c = id.x % HEAD_WIDTH;
    let score_base = (b * HEADS + h) * LENGTH * LENGTH;
    var query_grad = 0.0;
    var key_grad = 0.0;
    var value_grad = 0.0;
    for (var s = 0u; s <= t; s++) {
        query_grad += dscores[score_base + t * LENGTH + s] * qkv[packed_offset(b * LENGTH + s, 1u, h) + c];
    }
    for (var s = t; s < LENGTH; s++) {
        let att = score_base + s * LENGTH + t;
        key_grad += dscores[att] * qkv[packed_offset(b * LENGTH + s, 0u, h) + c];
        value_grad += prob[att] * dout[((b * LENGTH + s) * HEADS + h) * HEAD_WIDTH + c];
    }
    dqkv[packed_offset(row, 0u, h) + c] = query_grad;
    dqkv[packed_offset(row, 1u, h) + c] = key_grad;
    dqkv[packed_offset(row, 2u, h) + c] = value_grad;
}
