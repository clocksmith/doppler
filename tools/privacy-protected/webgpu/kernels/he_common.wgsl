// he_common.wgsl
//
// Shared declarations for the RNS-CKKS kernels. Every residue is one u32 below its modulus q < 2^32 (canonical:
// kernels read and write fully reduced values, so results equal Lattigo's bit for bit). WGSL has no 64-bit integers
// and no mul-high, so 32x32 -> 64-bit products are built from 16-bit halves, and products are reduced with
// Montgomery multiplication (R = 2^32). Multipliers that are known ahead (twiddles, keys, constants, plaintexts) are
// stored in Montgomery form, so one Montgomery multiplication returns the canonical product. The moduli here lie in
// (2^30, 2^32), so sums and Montgomery intermediates can exceed 2^32: carries are handled explicitly.
//
// This file is prepended to each kernel module by the host (he-device.js).

// ═══════════════════════════════════════════════════════════════════
// UNIFORMS (set per dispatch; offsets and strides are in u32 words)
// ═══════════════════════════════════════════════════════════════════

struct Uniforms {
    n: u32,         // ring degree N
    log_n: u32,     // log2(N)
    limbs: u32,     // limbs processed by this dispatch
    mod_base: u32,  // global modulus index of the first limb (Q moduli first, then P)
    a_off: u32,     // first word of operand a
    a_stride: u32,  // words between components of a (0 broadcasts one component)
    b_off: u32,
    b_stride: u32,
    o_off: u32,     // first word of the output
    o_stride: u32,
    c_off: u32,     // first word of the constant operand
    c_stride: u32,
    t: u32,         // kernel-specific: NTT butterfly distance, or conversion source count
    log_t: u32,     // kernel-specific: log2(t), or a global modulus index
    x0: u32,        // kernel-specific
    x1: u32,        // kernel-specific
}

// ═══════════════════════════════════════════════════════════════════
// BINDINGS
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> o: array<u32>;
@group(0) @binding(4) var<storage, read> c: array<u32>;
// Per global modulus: (q, -q^-1 mod 2^32, 2^64 mod q, N^-1 * 2^32 mod q).
@group(0) @binding(5) var<storage, read> mods: array<vec4<u32>>;
// Twiddles (Montgomery form, Lattigo's bit-reversed order, N per modulus) or other read-only tables.
@group(0) @binding(6) var<storage, read> tw: array<u32>;

override WORKGROUP_SIZE: u32 = 256u;

// ═══════════════════════════════════════════════════════════════════
// MODULAR ARITHMETIC
// ═══════════════════════════════════════════════════════════════════

// mul_wide returns (lo, hi) of the 64-bit product x * y.
fn mul_wide(x: u32, y: u32) -> vec2<u32> {
    let x0 = x & 0xffffu;
    let x1 = x >> 16u;
    let y0 = y & 0xffffu;
    let y1 = y >> 16u;
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    let mid = p01 + p10;
    let mid_carry = select(0u, 0x10000u, mid < p01);
    let lo = p00 + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < p00);
    let hi = p11 + (mid >> 16u) + mid_carry + lo_carry;
    return vec2<u32>(lo, hi);
}

// add_mod: (x + y) mod q for x, y < q < 2^32.
fn add_mod(x: u32, y: u32, q: u32) -> u32 {
    let s = x + y;
    if (s < x || s >= q) {
        return s - q;
    }
    return s;
}

// sub_mod: (x - y) mod q for x, y < q.
fn sub_mod(x: u32, y: u32, q: u32) -> u32 {
    if (x >= y) {
        return x - y;
    }
    return x - y + q;
}

// mont_mul: x * y * 2^-32 mod q for x, y < q (canonical result). qinv = -q^-1 mod 2^32.
fn mont_mul(x: u32, y: u32, q: u32, qinv: u32) -> u32 {
    let t = mul_wide(x, y);
    let m = t.x * qinv;
    let mq = mul_wide(m, q);
    // t.x + mq.x is 0 mod 2^32; it carries exactly when t.x != 0.
    let carry = select(0u, 1u, t.x != 0u);
    let s = t.y + mq.y;
    var over = s < t.y;
    let s2 = s + carry;
    over = over || (s2 < s);
    // The true value s2 + over * 2^32 is below 2q.
    if (over || s2 >= q) {
        return s2 - q;
    }
    return s2;
}

// reduce_below: x mod q for x < 3q (x itself below 2^32).
fn reduce_below(x: u32, q: u32) -> u32 {
    var r = x;
    if (r >= q) {
        r = r - q;
    }
    if (r >= q) {
        r = r - q;
    }
    return r;
}
