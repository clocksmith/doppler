// Offline diagnostic only. No runtime imports or qualification authority.
const bits = new DataView(new ArrayBuffer(4));
const UNIT = 2 ** -298;
const PRECISION = 1n << 24n;
const abs = (x) => x < 0n ? -x : x;

function f32Integer(value) {
  if (!Number.isFinite(value) || Math.fround(value) !== value) {
    throw new Error('Expected a finite, exactly represented F32 input');
  }
  bits.setFloat32(0, value, true);
  const word = bits.getUint32(0, true);
  const exponent = (word >>> 23) & 255;
  const mantissa = BigInt(word & 0x7fffff);
  const magnitude = exponent === 0 ? mantissa : (mantissa | (1n << 23n)) << BigInt(exponent - 1);
  return word >>> 31 ? -magnitude : magnitude;
}

export function dotEnvelope(input, weights, actual) {
  if (input.length !== weights.length || !input.length) throw new Error('Dot shape mismatch');
  const operations = BigInt(2 * input.length);
  if (operations >= PRECISION) throw new Error('gamma denominator is not positive');
  let exact = 0n;
  let absoluteSum = 0n;
  let forward = 0;
  let reverse = 0;
  for (let i = 0; i < input.length; i++) {
    for (const value of [input[i], weights[i]]) {
      if (value !== 0 && Math.abs(value) < 2 ** -126) throw new Error('Subnormal input needs an input-flush model');
    }
    const product = f32Integer(input[i]) * f32Integer(weights[i]);
    exact += product;
    absoluteSum += abs(product);
    forward = Math.fround(forward + Math.fround(input[i] * weights[i]));
    const j = input.length - 1 - i;
    reverse = Math.fround(reverse + Math.fround(input[j] * weights[j]));
  }
  const denominator = PRECISION - operations;
  // Absolute per-operation allowance for a flushed/subnormal result, propagated
  // through the same conservative rounding model. Not a driver certificate.
  const numerator = operations * absoluteSum + operations * PRECISION * (1n << 172n);
  if (Number(absoluteSum) * UNIT + Number(numerator) / Number(denominator) * UNIT >= 2 ** 127) {
    throw new Error('Overflow needs a separate arithmetic model');
  }
  const contains = (value) => {
    const error = abs((f32Integer(value) << 149n) - exact);
    return error * denominator <= numerator;
  };
  return {
    exact: Number(exact) * UNIT,
    error: Number(abs((f32Integer(actual) << 149n) - exact)) * UNIT,
    bound: Number(numerator) / Number(denominator) * UNIT,
    within: contains(actual), forward, reverse,
    forwardWithin: contains(forward), reverseWithin: contains(reverse),
    contains,
  };
}

export function halfValue(word) {
  if (!Number.isInteger(word) || word < 0 || word > 65535) throw new Error('Invalid F16 bits');
  const exponent = (word >>> 10) & 31;
  if (exponent === 31) throw new Error('Nonfinite F16 is outside this diagnostic');
  const fraction = word & 1023;
  const magnitude = exponent ? (1024 + fraction) * 2 ** (exponent - 25) : fraction * 2 ** -24;
  return word & 32768 ? -magnitude : magnitude;
}

export function nearestHalf(value) {
  if (!Number.isFinite(value) || Math.abs(value) >= 65520) throw new Error('F16 overflow/nonfinite input');
  const negative = value < 0 || Object.is(value, -0);
  const magnitude = Math.abs(value);
  let lo = 0;
  let hi = 31743;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (halfValue(mid) <= magnitude) lo = mid;
    else hi = mid - 1;
  }
  let result = lo;
  if (lo < 31743) {
    const midpoint = (halfValue(lo) + halfValue(lo + 1)) / 2;
    if (magnitude > midpoint || (magnitude === midpoint && lo % 2)) result++;
  }
  return result | (negative ? 32768 : 0);
}

export function cacheDecision(value, stored, referenceValue, referenceStored) {
  const ownCorrect = nearestHalf(value) === stored;
  const referenceCorrect = nearestHalf(referenceValue) === referenceStored;
  const different = stored !== referenceStored;
  const a = halfValue(stored);
  const b = halfValue(referenceStored);
  const adjacent = different && Math.sign(a) === Math.sign(b) && Math.abs(stored - referenceStored) === 1;
  const midpoint = adjacent ? (a + b) / 2 : null;
  return { ownCorrect, referenceCorrect, different, adjacent, midpoint,
    crossedMidpoint: midpoint === null ? null : Math.min(value, referenceValue) <= midpoint && Math.max(value, referenceValue) >= midpoint,
    upstreamStatus: 'unresolved-no-independent-input-bound' };
}
