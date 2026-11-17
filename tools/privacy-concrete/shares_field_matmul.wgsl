// Exact rows @ W modulo the prime 2^61 - 1 for the two-server shares route.
//
// WGSL has no 64-bit integers, so every field element arrives as four 16-bit limbs and every weight as an unsigned
// 16-bit value (the signed weight plus 32768). A limb times a weight fits exactly in 32 bits; the sum over the input
// width is carried into a second 32-bit word. The four limb sums are then rotated into place modulo the prime (a
// multiplication by 2^16k is a 61-bit rotation), added, and corrected for the weight offset.

struct Params {
  rows: u32,
  inputs: u32,
  outputs: u32,
  columnBegin: u32,
  columnCount: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

struct U64 {
  lo: u32,
  hi: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// rows * inputs * 4 limbs, least significant first.
@group(0) @binding(1) var<storage, read> limbs: array<u32>;
// outputs * inputs unsigned 16-bit weights, two per word, stored column by column (transposed).
@group(0) @binding(2) var<storage, read> weights: array<u32>;
// Per row: 32768 * (sum of the row) modulo the prime, as (lo, hi).
@group(0) @binding(3) var<storage, read> correction: array<u32>;
// rows * outputs results as (lo, hi), row-major.
@group(0) @binding(4) var<storage, read_write> results: array<u32>;

const PRIME_HI: u32 = 0x1FFFFFFFu;
const PRIME_LO: u32 = 0xFFFFFFFFu;

fn add64(a: U64, b: U64) -> U64 {
  let lo = a.lo + b.lo;
  let carry = select(0u, 1u, lo < a.lo);
  return U64(lo, a.hi + b.hi + carry);
}

fn accumulate(sum: U64, product: u32) -> U64 {
  let lo = sum.lo + product;
  return U64(lo, sum.hi + select(0u, 1u, lo < product));
}

// For values below 2^63: the residue below the prime.
fn reduce61(a: U64) -> U64 {
  var r = add64(U64(a.lo, a.hi & PRIME_HI), U64(a.hi >> 29u, 0u));
  if (r.hi > PRIME_HI || (r.hi == PRIME_HI && r.lo == PRIME_LO)) {
    // r - prime, where prime = 2^61 - 1: add one, then drop bit 61.
    let lo = r.lo + 1u;
    let hi = r.hi + select(0u, 1u, lo == 0u);
    r = U64(lo, hi & PRIME_HI);
  }
  return r;
}

// v * 2^16 modulo the prime, for v below 2^61.
fn rotate16(v: U64) -> U64 {
  let low = U64(v.lo << 16u, ((v.hi & 0x1FFFu) << 16u) | (v.lo >> 16u));
  return add64(low, U64(v.hi >> 13u, 0u));
}

// v * 2^32 modulo the prime, for v below 2^61.
fn rotate32(v: U64) -> U64 {
  let low = U64(0u, v.lo & PRIME_HI);
  return add64(low, U64((v.lo >> 29u) | (v.hi << 3u), 0u));
}

// v * 2^48 modulo the prime, for v below 2^61.
fn rotate48(v: U64) -> U64 {
  let low = U64(0u, (v.lo & 0x1FFFu) << 16u);
  return add64(low, U64((v.lo >> 13u) | (v.hi << 19u), v.hi >> 13u));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.y;
  if (id.x >= params.columnCount || row >= params.rows) {
    return;
  }
  let column = params.columnBegin + id.x;
  let weightBase = column * params.inputs;
  let limbBase = row * params.inputs * 4u;
  var s0 = U64(0u, 0u);
  var s1 = U64(0u, 0u);
  var s2 = U64(0u, 0u);
  var s3 = U64(0u, 0u);
  for (var i = 0u; i < params.inputs; i = i + 1u) {
    let index = weightBase + i;
    let weight = (weights[index >> 1u] >> ((index & 1u) * 16u)) & 0xFFFFu;
    let at = limbBase + i * 4u;
    s0 = accumulate(s0, limbs[at] * weight);
    s1 = accumulate(s1, limbs[at + 1u] * weight);
    s2 = accumulate(s2, limbs[at + 2u] * weight);
    s3 = accumulate(s3, limbs[at + 3u] * weight);
  }
  // Each limb sum is below inputs * 2^32, far below 2^61, so the rotations apply directly.
  var total = add64(add64(s0, rotate16(s1)), add64(rotate32(s2), rotate48(s3)));
  total = reduce61(total);
  // Remove the weight offset: subtract 32768 * (row sum), by adding its negation modulo the prime.
  let c = U64(correction[row * 2u], correction[row * 2u + 1u]);
  let negated = U64(PRIME_LO - c.lo, PRIME_HI - c.hi);
  total = reduce61(add64(total, negated));
  let out = (row * params.outputs + column) * 2u;
  results[out] = total.lo;
  results[out + 1u] = total.hi;
}
