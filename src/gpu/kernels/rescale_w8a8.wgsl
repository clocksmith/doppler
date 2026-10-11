// Two dispatches preserve the declared F32 activation-scale then weight-scale boundary.
override WORKGROUP_SIZE: u32 = 256u;
override INPUT_IS_I32: bool = true;
override SCALE_BY_ROW: bool = true;
struct Uniforms { rows: u32, columns: u32, inner: u32, row_words: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> packed_scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read_write> status: atomic<u32>;

fn integer_to_f32(bits: u32) -> u32 {
    let sign = bits & 0x80000000u;
    let magnitude = select(bits, 0u - bits, sign != 0u);
    if (magnitude == 0u) { return 0u; }
    var exponent = 31u - countLeadingZeros(magnitude);
    var significand = magnitude;
    if (exponent <= 23u) { significand <<= 23u - exponent; }
    else {
        let shift = exponent - 23u;
        significand = magnitude >> shift;
        let remainder = magnitude & ((1u << shift) - 1u);
        let halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (significand & 1u) != 0u)) { significand++; }
        if (significand == 0x1000000u) { significand >>= 1u; exponent++; }
    }
    return sign | ((exponent + 127u) << 23u) | (significand & 0x7fffffu);
}

fn scaled_f32(left: u32, right: u32) -> u32 {
    let magnitude = left & 0x7fffffffu;
    let sign = left & 0x80000000u;
    if (magnitude == 0u) { return sign; }
    var exponent = i32(magnitude >> 23u) + i32(right >> 23u) - 254;
    let a = (magnitude & 0x7fffffu) | 0x800000u;
    let b = (right & 0x7fffffu) | 0x800000u;
    let low_product = (a & 65535u) * (b & 65535u);
    let cross = (a >> 16u) * (b & 65535u) + (b >> 16u) * (a & 65535u);
    let low = low_product + (cross << 16u);
    let high = (a >> 16u) * (b >> 16u) + (cross >> 16u) + select(0u, 1u, low < low_product);
    var shift = 23u;
    if ((high & 32768u) != 0u) { shift = 24u; exponent++; }
    if (exponent < -127) { return sign; }
    // Only the largest subnormal bin can round up to the smallest normal value.
    let subnormal = exponent == -127;
    if (subnormal) { shift++; }
    var significand = (high << (32u - shift)) | (low >> shift);
    let remainder = low & ((1u << shift) - 1u);
    let halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (significand & 1u) != 0u)) { significand++; }
    if (subnormal) { return sign | select(0u, 0x800000u, significand == 0x800000u); }
    if (significand == 0x1000000u) { significand >>= 1u; exponent++; }
    if (exponent > 127) { atomicOr(&status, 16u); return 0u; }
    return sign | (u32(exponent + 127) << 23u) | (significand & 0x7fffffu);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= u.rows * u.columns) { return; }
    let scale_row = select(index % u.columns, index / u.columns, SCALE_BY_ROW);
    let scale_bits = packed_scales[scale_row * u.row_words];
    let scale_exponent = (scale_bits >> 23u) & 255u;
    if ((scale_bits & 0x80000000u) != 0u || scale_exponent == 0u || scale_exponent == 255u) {
        atomicOr(&status, 4u); output[index] = 0u; return;
    }
    var value_bits = input[index];
    if (INPUT_IS_I32) { value_bits = integer_to_f32(value_bits); }
    let exponent = (value_bits >> 23u) & 255u;
    if (exponent == 255u || (exponent == 0u && (value_bits & 0x7fffffffu) != 0u)) {
        atomicOr(&status, 16u); output[index] = 0u; return;
    }
    output[index] = scaled_f32(value_bits, scale_bits);
}
