// Direct nearest-even binary16 rounding of a normalized finite two-F32 pair.
// Each binary16 midpoint is exactly binary32; only at that midpoint can the
// normalized residual change the nearest half. No intermediate F32 collapse,
// pack2x16float rounding choice, or floating subnormal comparison is used.
fn pair_half_bits(value: vec2<f32>) -> u32 {
    let high_bits = bitcast<u32>(value.x);
    let sign = (high_bits >> 16u) & 0x8000u;
    let magnitude = high_bits & 0x7fffffffu;
    let exponent = magnitude >> 23u;
    let fraction = magnitude & 0x007fffffu;
    if (exponent == 255u) { return sign | select(0x7c00u, 0x7e00u, fraction != 0u); }
    if (exponent < 102u) { return sign; }
    if (exponent >= 143u) { return sign | 0x7c00u; }
    var shift = 13u;
    var significand = fraction;
    var base = (exponent - min(exponent, 112u)) << 10u;
    if (exponent < 113u) {
        shift = 126u - exponent;
        significand |= 0x00800000u;
        base = 0u;
    }
    let quotient = base | (significand >> shift);
    let remainder = significand & ((1u << shift) - 1u);
    let midpoint = 1u << (shift - 1u);
    let low_bits = bitcast<u32>(value.y);
    let low_nonzero = (low_bits & 0x7fffffffu) != 0u;
    let low_increases_magnitude = ((low_bits ^ high_bits) & 0x80000000u) == 0u;
    let tie_up = select((quotient & 1u) != 0u, low_increases_magnitude, low_nonzero);
    return sign | (quotient + u32(remainder > midpoint || (remainder == midpoint && tie_up)));
}
