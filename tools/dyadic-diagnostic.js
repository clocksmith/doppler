// Exact arithmetic helpers for offline diagnostic witnesses, never runtime execution.
const view = new DataView(new ArrayBuffer(8));

export function dyadic(value) {
  if (!Number.isFinite(value)) throw new Error('Finite dyadic input required');
  view.setFloat64(0, value, false);
  const bits = view.getBigUint64(0, false);
  const exponent = Number((bits >> 52n) & 2047n);
  const fraction = bits & ((1n << 52n) - 1n);
  return { n: ((bits >> 63n) ? -1n : 1n) * (exponent ? fraction + (1n << 52n) : fraction),
    e: exponent ? exponent - 1075 : -1074 };
}

export function add(a, b) {
  const e = Math.min(a.e, b.e);
  return { n: (a.n << BigInt(a.e - e)) + (b.n << BigInt(b.e - e)), e };
}

export function multiply(a, b) { return { n: a.n * b.n, e: a.e + b.e }; }

export function roundF32({ n, e }) {
  if (n === 0n) return 0;
  const negative = n < 0n;
  if (negative) n = -n;
  const length = n.toString(2).length;
  const shift = Math.max(length - 24, -149 - e, 0);
  let q = n >> BigInt(shift);
  if (shift > 0) {
    const remainder = n - (q << BigInt(shift));
    const halfway = 1n << BigInt(shift - 1);
    if (remainder > halfway || (remainder === halfway && q % 2n)) q++;
  }
  const magnitude = Number(q) * 2 ** (e + shift);
  if (!Number.isFinite(Math.fround(magnitude))) throw new Error('F32 overflow outside diagnostic');
  return negative ? -magnitude : magnitude;
}

export function fma(a, b, c) { return roundF32(add(multiply(dyadic(a), dyadic(b)), dyadic(c))); }

function integerSqrt(n) {
  if (n < 0n) throw new Error('Negative square root');
  if (n < 2n) return n;
  let x = 1n << BigInt(Math.ceil(n.toString(2).length / 2));
  for (;;) {
    const y = (x + n / x) >> 1n;
    if (y >= x) return x;
    x = y;
  }
}

// Outward enclosing inverse-square-root endpoints at a fixed 80 fraction bits.
// The integer square root removes dependence on Math.sqrt accuracy in the check.
export function inverseRootInterval(value) {
  const { n, e } = dyadic(value);
  if (n <= 0n || 160 - e < 0) throw new Error('Unsupported inverse-root range');
  const lower = integerSqrt((1n << BigInt(160 - e)) / n);
  return { lower: { n: lower, e: -80 }, upper: { n: lower + 1n, e: -80 } };
}

export function compare(a, b) { return add(a, { n: -b.n, e: b.e }).n; }

export function inverseRootCandidates(value) {
  const interval = inverseRootInterval(value);
  const nearest = roundF32(interval.lower);
  view.setFloat32(0, nearest, false);
  const center = view.getUint32(0, false);
  const exponent = interval.lower.n.toString(2).length - 1 - 80;
  const allowance = { n: 2n, e: exponent - 23 };
  const candidates = [];
  for (let delta = -5; delta <= 5; delta++) {
    view.setUint32(0, center + delta, false);
    const candidate = view.getFloat32(0, false);
    const c = dyadic(candidate);
    // Require both enclosing endpoints within the 2-ULP allowance.
    if (compare(add(c, allowance), interval.upper) >= 0n &&
        compare(add(c, { n: -allowance.n, e: allowance.e }), interval.lower) <= 0n) candidates.push(candidate);
  }
  if (!candidates.length) throw new Error('No certified inverse-root candidate');
  return candidates;
}
