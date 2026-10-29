// Diagnostic prototype: explicit IEEE binary16 nearest-even rounding.
// Each invocation owns an entire packed word, including odd range boundaries.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms { src_offset: u32, dst_offset: u32, count: u32, _pad0: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input_k: array<f32>;
@group(0) @binding(2) var<storage, read> input_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_k: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_v: array<u32>;

fn nearest_half(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    let sign = (bits >> 16u) & 0x8000u;
    let exponent = (bits >> 23u) & 255u;
    let fraction = bits & 0x7fffffu;
    if (exponent == 255u) { return sign | select(0x7c00u, 0x7e00u, fraction != 0u); }
    if (exponent < 102u) { return sign; }
    if (exponent >= 143u) { return sign | 0x7c00u; }
    var shift = 13u;
    var significant = fraction;
    var base = (exponent - min(exponent, 112u)) << 10u;
    if (exponent < 113u) { shift = 126u - exponent; significant |= 0x800000u; base = 0u; }
    let quotient = base | (significant >> shift);
    let remainder = significant & ((1u << shift) - 1u);
    let midpoint = 1u << (shift - 1u);
    let round_up = remainder > midpoint || (remainder == midpoint && (quotient & 1u) != 0u);
    return sign | (quotient + u32(round_up));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
    let pair = gid.x + gid.y * grid.x * WORKGROUP_SIZE;
    if (pair >= (u.count + (u.dst_offset & 1u) + 1u) / 2u) { return; }
    let word = u.dst_offset / 2u + pair;
    var packed_k = output_k[word];
    var packed_v = output_v[word];
    for (var lane = 0u; lane < 2u; lane++) {
        let destination = word * 2u + lane;
        if (destination >= u.dst_offset && destination - u.dst_offset < u.count) {
            let source = u.src_offset + destination - u.dst_offset;
            let shift = lane * 16u;
            let mask = ~(0xffffu << shift);
            packed_k = (packed_k & mask) | (nearest_half(input_k[source]) << shift);
            packed_v = (packed_v & mask) | (nearest_half(input_v[source]) << shift);
        }
    }
    output_k[word] = packed_k;
    output_v[word] = packed_v;
}
