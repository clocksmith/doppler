// Independent CPU-only boundary review; never imported by model execution.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const candidateUrl = new URL('../candidate-qknorm-deferred-half.wgsl', import.meta.url);
const candidateSha256 = sha256(await readFile(candidateUrl));
assert.equal(candidateSha256, '3ede1077dae8b9e69fc116b5a6bfcad165a68ea116a7ef269cc8220271b6633b');
const view = new DataView(new ArrayBuffer(4));
function bits(value) {
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}
function float(value) {
  view.setUint32(0, value, true);
  return view.getFloat32(0, true);
}

// Literal translation of the candidate's bitwise midpoint decision.
function deferHalfRounding(high, low) {
  const encoded = bits(high);
  const exponent = (encoded >>> 23) & 255;
  if (exponent < 102 || exponent > 142 || (bits(low) & 0x7fffffff) === 0) return high;
  const shift = exponent >= 113 ? 13 : 126 - exponent;
  const significand = (encoded & 0x007fffff) | 0x00800000;
  if ((significand & ((1 << shift) - 1)) !== (1 << (shift - 1))) return high;
  const sameSign = ((encoded ^ bits(low)) & 0x80000000) === 0;
  return float(sameSign ? encoded + 1 : encoded - 1);
}

// Independent reference: enumerate binary16 values, locate the two neighbors,
// compare their distances, and choose the even encoded significand on a tie.
// The virtual successor 65536 gives the IEEE overflow midpoint 65520.
function halfValue(encoded) {
  const exponent = encoded >>> 10;
  const significand = encoded & 1023;
  return exponent === 0 ? significand * 2 ** -24
    : (1024 + significand) * 2 ** (exponent - 25);
}
const positiveHalf = Array.from({ length: 0x7c01 }, (_, index) => halfValue(index));
function nearestHalf(value) {
  const negative = value < 0 || Object.is(value, -0);
  const magnitude = Math.abs(value);
  let lo = 0;
  let hi = 0x7c00;
  while (lo + 1 < hi) {
    const middle = (lo + hi) >>> 1;
    if (positiveHalf[middle] <= magnitude) lo = middle;
    else hi = middle;
  }
  const lowerDistance = magnitude - positiveHalf[lo];
  const upperDistance = positiveHalf[hi] - magnitude;
  const selected = lowerDistance < upperDistance ? lo
    : upperDistance < lowerDistance ? hi : (lo & 1) === 0 ? lo : hi;
  return selected | (negative ? 0x8000 : 0);
}

let comparisons = 0;
const mismatches = [];
for (let lower = 0; lower < 0x7c00; lower++) {
  const midpoint = (positiveHalf[lower] + positiveHalf[lower + 1]) / 2;
  const midpointBits = bits(midpoint);
  for (const sign of [1, -1]) {
    for (const adjacent of [-1, 0, 1]) {
      const high = sign * float(midpointBits + adjacent);
      const ulp = 2 ** (((bits(high) >>> 23) & 255) - 127 - 23);
      for (const residualSign of [-1, 0, 1]) {
        const low = residualSign * ulp / 4;
        // These values require at most 26 significant bits; addition in
        // binary64 is exact, so this does not repeat binary32 double rounding.
        const expected = nearestHalf(high + low);
        const actual = nearestHalf(deferHalfRounding(high, low));
        comparisons++;
        if (actual !== expected) mismatches.push({ lower, sign, adjacent, residualSign, high, low, expected, actual });
      }
    }
  }
}
const result = {
  candidateSha256,
  checkerSha256: sha256(await readFile(new URL(import.meta.url))),
  node: process.version,
  enumeration: { positiveMidpoints: 31744, signs: 2, adjacentF32Values: 3, residuals: 3 },
  residuals: 'negative quarter F32 ulp, zero, positive quarter F32 ulp',
  comparisons,
  mismatchCount: mismatches.length,
  mismatches: mismatches.slice(0, 16),
  scope: 'CPU helper translation and independent binary16 nearest-even reference; no GPU, normalization residual, geometry execution, or whole-model qualification',
};
console.log(JSON.stringify(result, null, 2));
assert.equal(mismatches.length, 0);
