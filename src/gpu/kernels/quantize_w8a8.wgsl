// Exact-rational symmetric W8A8 activation packing; one workgroup per token.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms { rows: u32, columns: u32, inner: u32, row_words: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
var<workgroup> maxima: array<u32, WORKGROUP_SIZE>;
var<workgroup> row_scale: u32;

fn rounded_scale(bits: u32) -> u32 {
    if (bits == 0u) { return 0x3f800000u; }
    let exponent = i32(bits >> 23u);
    let mantissa = (bits & 0x7fffffu) | 0x800000u;
    let shift = select(6u, 7u, mantissa < 16646144u);
    let numerator = mantissa << shift;
    let quotient = numerator / 127u;
    let remainder = numerator % 127u;
    let rounded = quotient + select(0u, 1u, remainder * 2u > 127u);
    let scale_exponent = exponent - i32(shift);
    if (scale_exponent <= 0) {
        atomicOr(&status, 4u);
        return 0x3f800000u;
    }
    return (u32(scale_exponent) << 23u) | (rounded & 0x7fffffu);
}

fn quantized_byte(bits: u32, scale: u32) -> u32 {
    let magnitude = bits & 0x7fffffffu;
    if (magnitude == 0u) { return 0u; }
    let exponent = magnitude >> 23u;
    if (exponent == 0u || exponent == 255u) { return 0u; }
    let delta = i32(exponent) - i32(scale >> 23u);
    if (delta <= -2) { return 0u; }
    if (delta > 7) { atomicOr(&status, 4u); return 0u; }
    var numerator = (magnitude & 0x7fffffu) | 0x800000u;
    var denominator = (scale & 0x7fffffu) | 0x800000u;
    if (delta < 0) { denominator = denominator << 1u; }
    else { numerator = numerator << u32(delta); }
    var quotient = numerator / denominator;
    let twice_remainder = (numerator % denominator) * 2u;
    if (twice_remainder > denominator || (twice_remainder == denominator && (quotient & 1u) != 0u)) {
        quotient += 1u;
    }
    let code = i32(min(quotient, 127u));
    return bitcast<u32>(select(code, -code, (bits & 0x80000000u) != 0u)) & 255u;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    if (row >= u.rows) { return; }
    let lane = local.x;
    var maximum = 0u;
    for (var k = lane; k < u.inner; k += WORKGROUP_SIZE) {
        let bits = input[row * u.inner + k] & 0x7fffffffu;
        let exponent = bits >> 23u;
        if (exponent == 255u) { atomicOr(&status, 1u); }
        else if (exponent == 0u && bits != 0u) { atomicOr(&status, 2u); }
        else { maximum = max(maximum, bits); }
    }
    maxima[lane] = maximum;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) { maxima[lane] = max(maxima[lane], maxima[lane + stride]); }
        workgroupBarrier();
    }
    if (lane == 0u) {
        row_scale = rounded_scale(maxima[0]);
        output[row * u.row_words] = row_scale;
    }
    workgroupBarrier();
    for (var word = lane; word < u.row_words - 1u; word += WORKGROUP_SIZE) {
        var packed = 0u;
        for (var byte = 0u; byte < 4u; byte++) {
            let k = word * 4u + byte;
            if (k < u.inner) { packed |= quantized_byte(input[row * u.inner + k], row_scale) << (byte * 8u); }
        }
        output[row * u.row_words + 1u + word] = packed;
    }
}
