// Native Q4_1 routed projection. One lane consumes a full 32-value block.
// Preserves original scale and minimum per weight; route reduction remains separate.
override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 4u;
override THREADS_PER_COL_GEMV: u32 = 32u;
override TILE_SIZE: u32 = 16u;
override BLOCK_VALUES: u32 = 32u;
override BLOCK_BYTES: u32 = 20u;
override QUANT_FORMAT: u32 = 1u;
struct Uniforms {
    M: u32, N: u32, K: u32, alpha: f32,
    blocks: u32, num_experts: u32, top_k: u32, input_mode: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> route_indices: array<u32>;
var<workgroup> partial: array<f32, WORKGROUP_SIZE>;
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
        let row = expert * u.N + n;
        for (var block = thread; block < u.blocks; block += THREADS_PER_COL_GEMV) {
            let offset = (row * u.blocks + block) * BLOCK_BYTES;
            let parameters = unpack2x16float(weights[offset / 4u]);
            let scale = parameters.x;
            let minimum = parameters.y;
            let base = token * u.K + block * BLOCK_VALUES;
            for (var j = 0u; j < 16u; j += 4u) {
                let packed_word = weights[(offset + 4u + j) / 4u];
                let packed = vec4<u32>(packed_word, packed_word >> 8u, packed_word >> 16u, packed_word >> 24u) & vec4<u32>(255u);
                let low = vec4<f32>(packed & vec4<u32>(15u)) * scale + vec4<f32>(minimum);
                let high = vec4<f32>(packed >> vec4<u32>(4u)) * scale + vec4<f32>(minimum);
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
