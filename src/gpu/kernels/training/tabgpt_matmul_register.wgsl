// Dense FP32 GEMM with a 4x4 result per invocation. Matrices in storage are
// row major. Each WGSL matrix column holds one logical row of a 4x4 tile.
override M: u32 = 4u;
override N: u32 = 4u;
override K: u32 = 4u;
override TRANSPOSE_A: bool = false;
override TRANSPOSE_B: bool = false;
@group(0) @binding(0) var<storage, read> left: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> right: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;

fn left_tile(row: u32, reduction: u32) -> mat4x4<f32> {
    if (TRANSPOSE_A) {
        let stride = M / 4u;
        let i = reduction * stride + row / 4u;
        return transpose(mat4x4<f32>(left[i], left[i + stride],
            left[i + 2u * stride], left[i + 3u * stride]));
    }
    let stride = K / 4u;
    let i = row * stride + reduction / 4u;
    return mat4x4<f32>(left[i], left[i + stride], left[i + 2u * stride], left[i + 3u * stride]);
}

fn right_tile(col: u32, reduction: u32) -> mat4x4<f32> {
    if (TRANSPOSE_B) {
        let stride = K / 4u;
        let i = col * stride + reduction / 4u;
        return mat4x4<f32>(right[i], right[i + stride], right[i + 2u * stride], right[i + 3u * stride]);
    }
    let stride = N / 4u;
    let i = reduction * stride + col / 4u;
    return mat4x4<f32>(right[i], right[i + stride], right[i + 2u * stride], right[i + 3u * stride]);
}

fn update_row(sum: vec4<f32>, a: vec4<f32>, b: mat4x4<f32>) -> vec4<f32> {
    if (TRANSPOSE_B) {
        return sum + vec4<f32>(dot(a, b[0]), dot(a, b[1]), dot(a, b[2]), dot(a, b[3]));
    }
    return sum + a.x * b[0] + a.y * b[1] + a.z * b[2] + a.w * b[3];
}

@compute @workgroup_size(16, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y * 4u;
    let col = id.x * 4u;
    if (row >= M || col >= N) { return; }
    var sums: mat4x4<f32>;
    for (var p = 0u; p < K; p += 4u) {
        let a = left_tile(row, p);
        let b = right_tile(col, p);
        sums[0] = update_row(sums[0], a[0], b);
        sums[1] = update_row(sums[1], a[1], b);
        sums[2] = update_row(sums[2], a[2], b);
        sums[3] = update_row(sums[3], a[3], b);
    }
    let stride = N / 4u;
    let i = row * stride + id.x;
    result[i] = sums[0];
    result[i + stride] = sums[1];
    result[i + 2u * stride] = sums[2];
    result[i + 3u * stride] = sums[3];
}
