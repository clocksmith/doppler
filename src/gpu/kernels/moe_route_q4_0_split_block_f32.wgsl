// Native Q4_0 separate gate/up projection. One lane consumes a full 32-value block.
// Preserves packed scales and quantization; output remains [routes, gate then up].
override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 4u;
override THREADS_PER_COL_GEMV: u32 = 32u;
override TILE_SIZE: u32 = 16u;
override BLOCK_VALUES: u32 = 32u;
override BLOCK_BYTES: u32 = 18u;
override QUANT_FORMAT: u32 = 0u;
struct Uniforms {
    M: u32, N: u32, K: u32, alpha: f32,
    blocks: u32, num_experts: u32, top_k: u32, input_mode: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> route_indices: array<u32>;
@group(0) @binding(5) var<storage, read> up_weights: array<u32>;
var<workgroup> partial: array<f32, WORKGROUP_SIZE>;
fn source_byte(offset: u32, up: bool) -> u32 {
    if (up) { return (up_weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u; }
    return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let tiles_n = (u.N + COLS_PER_WG - 1u) / COLS_PER_WG;
    let tile = group.x + group.y * grid.x;
    let m = tile / tiles_n;
    if (m >= u.M) { return; }
    let expert = route_indices[m];
    let token = select(m / u.top_k, m, u.input_mode == 1u);
    let column = lane / THREADS_PER_COL_GEMV;
    let thread = lane % THREADS_PER_COL_GEMV;
    let n = (tile % tiles_n) * COLS_PER_WG + column;
    var sum = 0.0f;
    if (n < u.N && expert < u.num_experts) {
        let row = expert * (u.N / 2u) + n % (u.N / 2u);
        let up = n >= u.N / 2u;
        for (var block = thread; block < u.blocks; block += THREADS_PER_COL_GEMV) {
            let offset = (row * u.blocks + block) * BLOCK_BYTES;
            let scale = unpack2x16float(source_byte(offset, up) | (source_byte(offset + 1u, up) << 8u)).x;
            let base = token * u.K + block * BLOCK_VALUES;
            for (var j = 0u; j < 16u; j += 4u) {
                let packed = vec4<u32>(source_byte(offset + 2u + j, up), source_byte(offset + 3u + j, up),
                    source_byte(offset + 4u + j, up), source_byte(offset + 5u + j, up));
                let low = (vec4<f32>(packed & vec4<u32>(15u)) - vec4<f32>(8.0)) * scale;
                let high = (vec4<f32>(packed >> vec4<u32>(4u)) - vec4<f32>(8.0)) * scale;
                let xLow = vec4<f32>(input[base + j], input[base + j + 1u], input[base + j + 2u], input[base + j + 3u]);
                let xHigh = vec4<f32>(input[base + j + 16u], input[base + j + 17u], input[base + j + 18u], input[base + j + 19u]);
                sum += dot(xLow, low) + dot(xHigh, high);
            }
        }
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = THREADS_PER_COL_GEMV / 2u; stride > 0u; stride /= 2u) {
        if (thread < stride) { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    if (thread == 0u && n < u.N) { output[m * u.N + n] = partial[lane] * u.alpha; }
}
