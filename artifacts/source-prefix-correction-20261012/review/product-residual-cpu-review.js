// Independent CPU review only; never imported by Doppler runtime.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const candidate = new URL('../candidate-range-product-rmsnorm.wgsl', import.meta.url);
const candidateSha256 = createHash('sha256').update(readFileSync(candidate)).digest('hex');
assert.equal(candidateSha256, 'a056cdd5a773d12d624a974d4c48bf57f9504bf4713496e44c5dfcf374fe9fc5');
const seed = 624901;
const samples = 500000;
const buffer = new ArrayBuffer(4);
const view = new DataView(buffer);
const number = bits => { view.setUint32(0, bits); return view.getFloat32(0); };
const bits = value => { view.setFloat32(0, value); return view.getUint32(0); };
const hex = value => `0x${value.toString(16).padStart(8, '0')}`;
const unsigned = value => value >>> 0;
const exponent = value => (value >>> 23) & 255;
const mantissa = value => (value & 0x7fffff) | (exponent(value) ? 0x800000 : 0);
const finite = value => exponent(value) !== 255;

// Candidate translation uses Number/U32 limbs, including WGSL wrapping shifts.
function add(a, b) {
  const low = unsigned(a[0] + b[0]);
  return [low, unsigned(a[1] + b[1] + Number(low < a[0]))];
}
function negate(a) { return [unsigned(-a[0]), unsigned(~a[1] + Number(a[0] === 0))]; }
function roundedShift(value, shift) {
  if (shift === 0) return value[0];
  if (shift >= 64) return 0;
  let quotient, above, tie;
  if (shift < 32) {
    quotient = unsigned((value[0] >>> shift) | (value[1] << (32 - shift)));
    const remainder = unsigned(value[0] & unsigned((1 << shift) - 1));
    const halfway = unsigned(1 << (shift - 1));
    above = remainder > halfway; tie = remainder === halfway;
  } else if (shift === 32) {
    quotient = value[1]; above = value[0] > 0x80000000; tie = value[0] === 0x80000000;
  } else {
    const highShift = shift - 32;
    quotient = value[1] >>> highShift;
    const remainder = unsigned(value[1] & unsigned((1 << highShift) - 1));
    const halfway = unsigned(1 << (highShift - 1));
    above = remainder > halfway || (remainder === halfway && value[0] !== 0);
    tie = remainder === halfway && value[0] === 0;
  }
  return unsigned(quotient + Number(above || (tie && (quotient & 1) !== 0)));
}
function scaled(magnitude, power, negative) {
  const sign = negative ? 0x80000000 : 0;
  if (!(magnitude[0] || magnitude[1])) return sign;
  const top = magnitude[1] ? 63 - Math.clz32(magnitude[1]) : 31 - Math.clz32(magnitude[0]);
  let unbiased = power + top;
  let output;
  if (unbiased >= -126) {
    let significand = top > 23 ? roundedShift(magnitude, top - 23) : unsigned(magnitude[0] << (23 - top));
    if (significand >= 0x1000000) { significand >>>= 1; unbiased++; }
    output = unsigned(((unbiased + 127) << 23) | (significand & 0x7fffff));
  } else {
    const shift = power + 149;
    output = shift >= 0 ? unsigned(magnitude[0] << shift) : roundedShift(magnitude, -shift);
  }
  return unsigned(sign | output);
}
let limbChecks = 0;
function translated(a, b, p) {
  const ae = exponent(a), be = exponent(b), pe = exponent(p);
  if (ae === 255 || be === 255 || pe === 255 || !(a & 0x7fffffff) || !(b & 0x7fffffff)) return 0;
  const am = mantissa(a), bm = mantissa(b), pm = mantissa(p);
  const al = am & 65535, ah = am >>> 16, bl = bm & 65535, bh = bm >>> 16;
  const cross = al * bh + ah * bl;
  const exact = add([al * bl, ah * bh], [unsigned(cross << 16), cross >>> 16]);
  assert.equal(BigInt(exact[0]) + (BigInt(exact[1]) << 32n), BigInt(am) * BigInt(bm));
  limbChecks++;
  const power = Math.max(ae, 1) + Math.max(be, 1) - 300;
  const productNegative = unsigned(a ^ b) >= 0x80000000;
  if (!(p & 0x7fffffff)) return scaled(exact, power, productNegative);
  const shift = Math.max(pe, 1) + 150 - Math.max(ae, 1) - Math.max(be, 1);
  if (shift >= 64) return unsigned(p ^ 0x80000000);
  if (shift < 0) return 0;
  const rounded = shift === 0 ? [pm, 0] : shift < 32
    ? [unsigned(pm << shift), pm >>> (32 - shift)] : [0, unsigned(pm << (shift - 32))];
  const residual = add(exact, negate(rounded));
  const negative = residual[1] >= 0x80000000;
  return scaled(negative ? negate(residual) : residual, power, negative !== productNegative);
}

// Independent reference: exact signed dyadic integers, not limb arithmetic.
function dyadic(value) {
  const coefficient = BigInt(mantissa(value)) * ((value >>> 31) ? -1n : 1n);
  return [coefficient, Math.max(exponent(value), 1) - 150];
}
function roundInteger(value, shift) {
  if (shift <= 0) return value << BigInt(-shift);
  const divisor = 1n << BigInt(shift);
  const q = value / divisor, r = value % divisor;
  return q + BigInt(2n * r > divisor || (2n * r === divisor && (q & 1n) !== 0n));
}
function reference(a, b, p) {
  const [ac, ax] = dyadic(a), [bc, bx] = dyadic(b), [pc, px] = dyadic(p);
  const power = Math.min(ax + bx, px);
  const exact = (ac * bc << BigInt(ax + bx - power)) - (pc << BigInt(px - power));
  if (exact === 0n) return 0;
  const negative = exact < 0n, magnitude = negative ? -exact : exact;
  const top = magnitude.toString(2).length - 1;
  const targetPower = Math.max(-149, power + top - 23);
  let significand = roundInteger(magnitude, targetPower - power);
  let outputPower = targetPower;
  if (significand >= 0x1000000n) { significand >>= 1n; outputPower++; }
  const encoded = significand < 0x800000n ? Number(significand)
    : ((outputPower + 150) * 0x800000 + Number(significand - 0x800000n));
  return unsigned((negative ? 0x80000000 : 0) | encoded);
}

let cases = 0;
const lanes = {}, failures = [];
function check(a, b, p, lane) {
  const actual = translated(a, b, p), expected = reference(a, b, p);
  // An exact zero residual has no application-sign contract; retain underflow zero signs.
  const equal = actual === expected || (expected === 0 && (actual & 0x7fffffff) === 0);
  cases++; lanes[lane] = (lanes[lane] ?? 0) + 1;
  if (!equal && failures.length < 16) failures.push({ a: hex(a), b: hex(b), p: hex(p), actual: hex(actual), expected: hex(expected), lane });
  return { a: hex(a), b: hex(b), p: hex(p), actual: hex(actual), expected: hex(expected), equal };
}
const midpointCases = [];
for (const a of [0x3f800001, 0x3f800003]) {
  for (const sa of [0, 0x80000000]) for (const sb of [0, 0x80000000]) {
    const aa = unsigned(a ^ sa), bb = unsigned(0x3fc00000 ^ sb);
    midpointCases.push(check(aa, bb, bits(number(aa) * number(bb)), 'midpoint-sign'));
  }
}
const regressions = [
  [0x00800000, 0x00800000, 0x00000001],
  [0x360188cd, 0x0411747a, 0x00000934],
  [0x00000001, 0x7f000000, 0],
  [0x00000001, 0x7f7fffff, 0],
];
for (const [a, b, p] of regressions) {
  for (const sa of [0, 0x80000000]) for (const sb of [0, 0x80000000]) {
    check(unsigned(a ^ sa), unsigned(b ^ sb), unsigned(p ^ sa ^ sb), 'reported-boundary-sign');
  }
}
let state = seed;
function randomU32() {
  state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
  return unsigned(state);
}
let excludedNonfiniteOperands = 0, excludedOverflow = 0;
for (let i = 0; i < samples; i++) {
  const a = randomU32(), b = randomU32();
  if (!finite(a) || !finite(b)) { excludedNonfiniteOperands++; continue; }
  const exactProduct = number(a) * number(b);
  const p = bits(exactProduct);
  if (!finite(p)) { excludedOverflow++; continue; }
  check(a, b, p, 'nearest');
  if (number(p) !== exactProduct) {
    // Other enclosing F32 value; binary64 represents the exact 24x24 product.
    const positive = exactProduct > 0;
    const other = (p & 0x7fffffff) === 0
      ? unsigned((positive ? 0 : 0x80000000) | 1)
      : unsigned(p + ((number(p) < exactProduct) === positive ? 1 : -1));
    if (finite(other)) check(a, b, other, 'other-adjacent');
  }
  if (exponent(a) === 0 || exponent(b) === 0 || exponent(p) === 0) {
    check(a, b, unsigned((a ^ b) & 0x80000000), 'synthetic-ftz');
  }
}
const report = {
  schema: 'doppler-independent-product-residual-cpu-review-v1',
  candidateSha256,
  checkerSha256: createHash('sha256').update(readFileSync(new URL(import.meta.url))).digest('hex'),
  runtime: process.version, platform: process.platform, architecture: process.arch,
  seed, prng: 'xorshift32 (13,17,5)', samples, cases, limbChecks, lanes,
  excludedNonfiniteOperands, excludedOverflow, midpointCases, failures,
  scope: 'CPU translation and exact dyadic reference; no shader execution, GPU-product observations, full-model accuracy, performance or promotion claim',
};
console.log(JSON.stringify(report, null, 2));
assert.equal(failures.length, 0, 'Independent CPU review found mismatches');
