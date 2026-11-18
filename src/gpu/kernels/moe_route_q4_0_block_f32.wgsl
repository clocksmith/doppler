// Native Q4_0 routed projection. One lane consumes a full 32-value block via whole-word u32 loads.
// Tuned for 256-thread / 8-column Metal & Vulkan threadgroups while preserving exact block accumulation.
override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 8u;
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
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> route_indices: array<u32>;
var<workgroup> partial: array<f32, WORKGROUP_SIZE>;
fn unpack_bytes(word: u32) -> vec4<u32> {
    return vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(255u);
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
        let row = expert * u.N + n;
        for (var block = thread; block < u.blocks; block += THREADS_PER_COL_GEMV) {
            let offset = (row * u.blocks + block) * BLOCK_BYTES;
            let w_base = offset / 4u;
            let half_aligned = (offset & 2u) != 0u;
            let w0 = weights[w_base];
            let w1 = weights[w_base + 1u];
            let w2 = weights[w_base + 2u];
            let w3 = weights[w_base + 3u];
            let w4 = weights[w_base + 4u];
            let scale = select(unpack2x16float(w0).x, unpack2x16float(w0).y, half_aligned);
            let q0 = select((w0 >> 16u) | (w1 << 16u), w1, half_aligned);
            let q1 = select((w1 >> 16u) | (w2 << 16u), w2, half_aligned);
            let q2 = select((w2 >> 16u) | (w3 << 16u), w3, half_aligned);
            let q3 = select((w3 >> 16u) | (w4 << 16u), w4, half_aligned);
            let vec_base = (token * u.K + block * BLOCK_VALUES) / 4u;
            let p0 = unpack_bytes(q0);
            sum += dot(input[vec_base], (vec4<f32>(p0 & vec4<u32>(15u)) - vec4<f32>(8.0)) * scale)
                + dot(input[vec_base + 4u], (vec4<f32>(p0 >> vec4<u32>(4u)) - vec4<f32>(8.0)) * scale);
            let p1 = unpack_bytes(q1);
            sum += dot(input[vec_base + 1u], (vec4<f32>(p1 & vec4<u32>(15u)) - vec4<f32>(8.0)) * scale)
                + dot(input[vec_base + 5u], (vec4<f32>(p1 >> vec4<u32>(4u)) - vec4<f32>(8.0)) * scale);
            let p2 = unpack_bytes(q2);
            sum += dot(input[vec_base + 2u], (vec4<f32>(p2 & vec4<u32>(15u)) - vec4<f32>(8.0)) * scale)
                + dot(input[vec_base + 6u], (vec4<f32>(p2 >> vec4<u32>(4u)) - vec4<f32>(8.0)) * scale);
            let p3 = unpack_bytes(q3);
            sum += dot(input[vec_base + 3u], (vec4<f32>(p3 & vec4<u32>(15u)) - vec4<f32>(8.0)) * scale)
                + dot(input[vec_base + 7u], (vec4<f32>(p3 >> vec4<u32>(4u)) - vec4<f32>(8.0)) * scale);
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
