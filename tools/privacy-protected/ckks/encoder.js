// CKKS slot encoding for the 64-slot layout (logSlots = 6), as Lattigo's encoder does it.
//
// encodeCompact is Lattigo's arbitrary-precision path (64-bit big-float special IFFT, then round(scale * x) into
// every limb, then the sparse NTT): deterministic on every machine, and the compiler's plaintext constants come out
// word for word as Lattigo's. A 64-slot plaintext in the NTT domain is constant on blocks of N/128 coefficients,
// so one limb is 128 words (the "compact" form the WebGPU kernels read). decode is the float64 inverse.

import { BigFloat, Complex, ComplexMultiplier, cos, newFloat, pi } from './bigfloat.js';
import { LOG_SLOTS, SLOTS } from '../plan.js';
import { ringOf } from './params.js';
import { bitReverse, invmod, mulmod } from './ring.js';

export const COMPACT_WORDS = 2 * SLOTS;
const PREC = 64;

function rotGroup(m, count) {
  const out = [];
  let f = 1;
  for (let i = 0; i < count; i++) {
    out.push(f);
    f = (f * 5) % m;
  }
  return out;
}

// GetRootsBigComplex(m, 64), computed only for the indices the 64-slot transform touches.
class BigRoots {
  constructor(m) {
    this.m = m;
    this.quarm = m >> 2;
    const e2ipi = newFloat(2, PREC);
    e2ipi.mul(e2ipi, pi(PREC));
    e2ipi.quo(e2ipi, newFloat(m, PREC));
    this.e2ipi = e2ipi;
    this.cosCache = new Map();
    this.cache = new Map();
  }

  c(i) {
    // roots[i].Real() for 1 <= i < quarm: Cos(e2ipi * i).
    let v = this.cosCache.get(i);
    if (!v) {
      const angle = new BigFloat(PREC).mul(this.e2ipi, newFloat(i, PREC));
      v = cos(angle);
      this.cosCache.set(i, v);
    }
    return v;
  }

  base(i) {
    // roots[i] for 0 <= i <= quarm.
    if (i === 0) return new Complex(newFloat(1, PREC), newFloat(0, PREC));
    if (i === this.quarm) return new Complex(newFloat(0, PREC), newFloat(1, PREC));
    return new Complex(this.c(i), new BigFloat().set(this.c(this.quarm - i)));
  }

  get(k) {
    let r = this.cache.get(k);
    if (r) return r;
    const q = this.quarm;
    if (k === this.m || k === 0) r = this.base(0);
    else if (k <= q) r = this.base(k);
    else if (k <= 2 * q) {
      const s = this.base(q - (k - q));
      r = new Complex(new BigFloat().neg_(s.re), new BigFloat().set(s.im));
    } else if (k <= 3 * q) {
      const s = this.base(k - 2 * q);
      r = new Complex(new BigFloat().neg_(s.re), new BigFloat().neg_(s.im));
    } else {
      const s = this.base(q - (k - 3 * q));
      r = new Complex(new BigFloat().set(s.re), new BigFloat().neg_(s.im));
    }
    this.cache.set(k, r);
    return r;
  }
}

const rootsByM = new Map();

function bigRoots(m) {
  if (!rootsByM.has(m)) rootsByM.set(m, new BigRoots(m));
  return rootsByM.get(m);
}

function bitReverseInPlace(values, n) {
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    if (i < j) [values[i], values[j]] = [values[j], values[i]];
  }
}

// SpecialIFFTArbitrary(values, N = slots, M = 2 * ringN).
function specialIFFTArbitrary(values, n, m) {
  const roots = bigRoots(m);
  const rg = rotGroup(m, n);
  const u = new Complex();
  const v = new Complex();
  const cMul = new ComplexMultiplier();
  const logN = Math.log2(n);
  const logM = Math.log2(m);
  for (let loglen = logN; loglen > 0; loglen--) {
    const len = 1 << loglen;
    const lenh = len >> 1;
    const lenq = len << 2;
    const logGap = logM - 2 - loglen;
    const mask = lenq - 1;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < lenh; j++) {
        u.add(values[i + j], values[i + j + lenh]);
        v.sub(values[i + j], values[i + j + lenh]);
        cMul.mul(v, roots.get((lenq - (rg[j] & mask)) * 2 ** logGap), v);
        values[i + j].set(u);
        values[i + j + lenh].set(v);
      }
    }
  }
  const nBig = new BigFloat().setInt64(n);
  for (const x of values) {
    x.re.quo(x.re, nBig);
    x.im.quo(x.im, nBig);
  }
  bitReverseInPlace(values, n);
}

// The 2*slots integer coefficients (before reduction): round(scale * x), real parts then imaginary parts.
export function quantize(values, scale) {
  const n = SLOTS;
  const half = new BigFloat().setFloat64(0.5);
  const xFlo = new BigFloat();
  const out = new Array(2 * n);
  const one = (x) => {
    xFlo.mul(scale, x);
    if (x.sign() < 0) xFlo.sub(xFlo, half);
    else xFlo.add(xFlo, half);
    return xFlo.toBigInt();
  };
  for (let i = 0; i < n; i++) out[i] = one(values[i].re);
  for (let i = 0; i < n; i++) out[i + n] = one(values[i].im);
  return out;
}

// The arbitrary-precision embedding of up to 64 real values: 128 big integers.
export function embed(values, scale, m) {
  if (values.length > SLOTS) throw new Error(`at most ${SLOTS} values`);
  const buf = Array.from({ length: SLOTS }, (_, i) => new Complex(newFloat(values[i] ?? 0, PREC), newFloat(0, PREC)));
  specialIFFTArbitrary(buf, SLOTS, m);
  return quantize(buf, scale);
}

// Compact NTT words for limbs 0..level (limb-major, 128 words per limb).
export function encodeCompact(p, values, scale, level) {
  const ring = ringOf(p);
  const ints = embed(values, scale, 2 * p.n);
  const out = new Uint32Array((level + 1) * COMPACT_WORDS);
  const row = new Float64Array(COMPACT_WORDS);
  for (let i = 0; i <= level; i++) {
    const mod = ring.Q[i];
    const qb = BigInt(mod.q);
    for (let j = 0; j < COMPACT_WORDS; j++) {
      const r = ints[j] % qb;
      row[j] = Number(r < 0n ? r + qb : r);
    }
    mod.ntt(row, COMPACT_WORDS);
    out.set(row, i * COMPACT_WORDS);
  }
  return out;
}

// A full plaintext at `level` in the NTT domain: (level + 1) x N words, each limb its compact words repeated.
export function expandCompact(p, compact, level) {
  const n = p.n;
  const block = n / COMPACT_WORDS;
  const out = new Uint32Array((level + 1) * n);
  for (let i = 0; i <= level; i++) {
    for (let b = 0; b < COMPACT_WORDS; b++) out.fill(compact[i * COMPACT_WORDS + b], i * n + b * block, i * n + (b + 1) * block);
  }
  return out;
}

// --- decoding (float64) ---

function rootsDouble(m) {
  const out = new Array(m + 1);
  for (let k = 0; k <= m; k++) out[k] = [Math.cos((2 * Math.PI * k) / m), Math.sin((2 * Math.PI * k) / m)];
  return out;
}

const decodeRoots = new Map();

// SpecialFFTDouble on [re, im] pairs.
function specialFFTDouble(values, n, m) {
  if (!decodeRoots.has(m)) {
    // Only indices (rotGroup & mask) << logGap are used; precompute lazily via a sparse map.
    decodeRoots.set(m, new Map());
  }
  const cache = decodeRoots.get(m);
  const root = (k) => {
    let r = cache.get(k);
    if (!r) {
      r = [Math.cos((2 * Math.PI * k) / m), Math.sin((2 * Math.PI * k) / m)];
      cache.set(k, r);
    }
    return r;
  };
  const rg = rotGroup(m, n);
  bitReverseInPlace(values, n);
  const logN = Math.log2(n);
  const logM = Math.log2(m);
  for (let loglen = 1; loglen <= logN; loglen++) {
    const len = 1 << loglen;
    const lenh = len >> 1;
    const lenq = len << 2;
    const logGap = logM - 2 - loglen;
    const mask = lenq - 1;
    for (let i = 0; i < n; i += len) {
      for (let j = 0, k = i; j < lenh; j++, k++) {
        const [wr, wi] = root((rg[j] & mask) * 2 ** logGap);
        const [vr, vi] = values[k + lenh];
        const tr = vr * wr - vi * wi;
        const ti = vr * wi + vi * wr;
        const [ur, ui] = values[k];
        values[k] = [ur + tr, ui + ti];
        values[k + lenh] = [ur - tr, ui - ti];
      }
    }
  }
}

// decode: plaintext limbs 0..level in the coefficient domain (after INTT) -> 64 slot values (real parts).
export function decodeCoeffs(p, limbs, level, scale) {
  const n = p.n;
  const gap = n / COMPACT_WORDS;
  const q = p.q.slice(0, level + 1).map(BigInt);
  const Q = q.reduce((a, b) => a * b, 1n);
  const crt = q.map((qi, i) => {
    const hat = Q / qi;
    return hat * BigInt(invmod(Number(hat % qi), p.q[i]));
  });
  const halfQ = Q >> 1n;
  const coeff = (idx) => {
    let x = 0n;
    for (let i = 0; i <= level; i++) x += BigInt(limbs[i][idx]) * crt[i];
    x %= Q;
    if (x > halfQ) x -= Q;
    return Number(x) / scale;
  };
  const vals = [];
  for (let j = 0; j < SLOTS; j++) vals.push([coeff(j * gap), coeff((j + SLOTS) * gap)]);
  specialFFTDouble(vals, SLOTS, 2 * n);
  return vals.map((v) => v[0]);
}

export { LOG_SLOTS, mulmod };
