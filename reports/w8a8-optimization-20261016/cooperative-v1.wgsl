// Cooperative exact I32 reduction: neighboring lanes read neighboring packed
// words, rather than striding between entire weight rows. No subgroup dependence.
requires packed_4x8_integer_dot_product;
override WORKGROUP_SIZE: u32 = 64u;
override LANES_PER_OUTPUT: u32 = 16u;
struct Uniforms { rows: u32, columns: u32, inner: u32, row_words: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> activations: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<i32>;
@group(0) @binding(4) var<storage, read_write> status: atomic<u32>;
var<workgroup> sums: array<i32, WORKGROUP_SIZE>;
var<workgroup> nonzero_a: array<u32, WORKGROUP_SIZE>;
var<workgroup> nonzero_w: array<u32, WORKGROUP_SIZE>;

fn signed_byte(word: u32, byte: u32) -> i32 {
    return bitcast<i32>(word << ((3u - byte) * 8u)) >> 24u;
}

fn valid_scale(bits: u32) -> bool {
    let exponent = (bits >> 23u) & 255u;
    return (bits & 0x80000000u) == 0u && exponent > 0u && exponent < 255u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let lane = tid % LANES_PER_OUTPUT;
    let column = group.x * (WORKGROUP_SIZE / LANES_PER_OUTPUT) + tid / LANES_PER_OUTPUT;
    let row = group.y;
    let in_bounds = row < u.rows && column < u.columns;
    let a_base = row * u.row_words;
    let w_base = column * u.row_words;
    var sum = 0;
    var a_nonzero = 0u;
    var w_nonzero = 0u;
    var flags = 0u;
    if (in_bounds) {
        for (var word = lane; word < u.row_words - 1u; word += LANES_PER_OUTPUT) {
            let left = activations[a_base + 1u + word];
            let right = weights[w_base + 1u + word];
            var valid = true;
            for (var byte = 0u; byte < 4u; byte++) {
                let a = signed_byte(left, byte);
                let w = signed_byte(right, byte);
                let padding = word * 4u + byte >= u.inner;
                if (a == -128 || w == -128 || (padding && (a != 0 || w != 0))) { valid = false; }
            }
            if (!valid) { flags |= 8u; }
            else { sum += dot4I8Packed(left, right); }
            a_nonzero |= left;
            w_nonzero |= right;
        }
    }
    sums[tid] = sum;
    nonzero_a[tid] = a_nonzero;
    nonzero_w[tid] = w_nonzero;
    workgroupBarrier();
    // The absolute sum is bounded by K * 127^2, including intermediate sums.
    for (var offset = LANES_PER_OUTPUT / 2u; offset > 0u; offset /= 2u) {
        if (lane < offset) {
            sums[tid] += sums[tid + offset];
            nonzero_a[tid] |= nonzero_a[tid + offset];
            nonzero_w[tid] |= nonzero_w[tid + offset];
        }
        workgroupBarrier();
    }
    if (in_bounds && lane == 0u) {
        let a_scale = activations[a_base];
        let w_scale = weights[w_base];
        if (!valid_scale(a_scale) || !valid_scale(w_scale)
            || (nonzero_a[tid] == 0u && a_scale != 0x3f800000u)
            || (nonzero_w[tid] == 0u && w_scale != 0x3f800000u)) { flags |= 4u; }
        output[row * u.columns + column] = sums[tid];
    }
    if (flags != 0u) { atomicOr(&status, flags); }
}
