// Portable signed-byte dot products over the same row-interleaved W8A8 ABI.
override WORKGROUP_SIZE: u32 = 64u;
struct Uniforms { rows: u32, columns: u32, inner: u32, row_words: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> activations: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<i32>;
@group(0) @binding(4) var<storage, read_write> status: atomic<u32>;

fn signed_byte(word: u32, byte: u32) -> i32 {
    return bitcast<i32>(word << ((3u - byte) * 8u)) >> 24u;
}

fn dot_bytes(left: u32, right: u32) -> i32 {
    var sum = 0;
    for (var byte = 0u; byte < 4u; byte++) { sum += signed_byte(left, byte) * signed_byte(right, byte); }
    return sum;
}

fn valid_scale(bits: u32) -> bool {
    let exponent = (bits >> 23u) & 255u;
    return (bits & 0x80000000u) == 0u && exponent > 0u && exponent < 255u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= u.rows * u.columns) { return; }
    let row = index / u.columns;
    let column = index % u.columns;
    let a_base = row * u.row_words;
    let w_base = column * u.row_words;
    if (!valid_scale(activations[a_base]) || !valid_scale(weights[w_base])) { atomicOr(&status, 4u); }
    var sum = 0;
    var nonzero_a = 0u;
    var nonzero_w = 0u;
    for (var word = 0u; word < u.row_words - 1u; word++) {
        let left = activations[a_base + 1u + word];
        let right = weights[w_base + 1u + word];
        var valid = true;
        for (var byte = 0u; byte < 4u; byte++) {
            let a = signed_byte(left, byte);
            let w = signed_byte(right, byte);
            let padding = word * 4u + byte >= u.inner;
            if (a == -128 || w == -128 || (padding && (a != 0 || w != 0))) { valid = false; }
        }
        if (!valid) { atomicOr(&status, 8u); }
        else { sum += dot_bytes(left, right); }
        nonzero_a |= left;
        nonzero_w |= right;
    }
    if ((nonzero_a == 0u && activations[a_base] != 0x3f800000u)
        || (nonzero_w == 0u && weights[w_base] != 0x3f800000u)) { atomicOr(&status, 4u); }
    output[index] = sum;
}
