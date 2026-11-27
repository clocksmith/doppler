// Compiles the tiny block's encrypted circuit into a flat RNS program for the WebGPU evaluator
// (doppler.protected-gpu-program/v1: an op list, a constant blob, and steps: a prefill, then one decode step per
// position).
//
// The circuit is the one the removed Go/Lattigo CPU evaluator ran, and the compiler reproduces Lattigo v6.2.0's
// choices exactly: its CKKS evaluator (scale matching, integer scale ratios, scalar and plaintext constants), its
// Paterson-Stockmeyer polynomial evaluation in the Chebyshev basis, and its 128-bit scale arithmetic (ckks/bigfloat.js).
// A program depends only on the public model, the profile and the session shape, never on keys or inputs; the
// compiler holds no key at all. fixtures/go-golden.json pins the digests of the programs the Go recorder produced.

import { createHash } from 'node:crypto';
import { BigFloat } from './ckks/bigfloat.js';
import { encodeCompact } from './ckks/encoder.js';
import { lookupProfile, scaleOf } from './ckks/params.js';
import { bigMod } from './ckks/ring.js';
import { loadTriageModel, QUADRATIC_TERM, queryAt } from './triage-model.js';
import { CONTEXT as TRIAGE_CONTEXT } from './triage-task.js';
import {
  CONTEXT, DIM, HIDDEN, LOG_SLOTS, PROTOCOL_ID, SLOTS, VOCAB, loadModel, newPlan, replicate,
} from './plan.js';

export const PROGRAM_SCHEMA = 'doppler.protected-gpu-program/v1';
const ENCODING_PRECISION = 53; // Lattigo's EncodingPrecision for these parameters (scalars are rounded to it)

// --- Lattigo scale helpers (rlwe.Scale: 128-bit big floats) ---

const sMul = (a, b) => new BigFloat().mul(a, b);
const sDiv = (a, b) => new BigFloat().quo(a, b);
const sCmp = (a, b) => a.cmp(b);
const sMax = (a, b) => (a.cmp(b) < 0 ? b : a);

// bignum.ToComplex(x, 53)[0] for a float64, an int, or a 128-bit coefficient.
function toEncoding(x) {
  const z = new BigFloat(ENCODING_PRECISION);
  if (x instanceof BigFloat) return z.set(x);
  if (Number.isInteger(x)) return z.setInt64(x);
  return z.setFloat64(x);
}

// bigComplexToRNSScalar's real part: round(c * scale) as a big integer (round half away from zero at 128 bits).
function roundScaled(c, scale) {
  const r = new BigFloat().mul(c, scale);
  const sgn = c.sign();
  if (sgn > 0) r.add(r, new BigFloat().setFloat64(0.5));
  else if (sgn < 0) r.sub(r, new BigFloat().setFloat64(0.5));
  return r.toBigInt();
}

function bitsLen(x) {
  return x === 0 ? 0 : 32 - Math.clz32(x);
}

// --- the recorder: a Lattigo-equivalent evaluator that emits RNS ops instead of computing ---

class Recorder {
  constructor(p) {
    this.p = p;
    this.regs = [];
    this.ops = [];
    this.blob = [];
    this.blobLen = 0;
    this.counts = {};
    this.gal = new Set();
    this.from = '';
  }

  newReg(deg, level) {
    this.regs.push({ deg, level });
    return this.regs.length - 1;
  }

  words(v) {
    const off = this.blobLen;
    this.blob.push(v);
    this.blobLen += v.length;
    return off;
  }

  emit(op) {
    const o = { op: op.op, out: 0, a: op.a ?? -1, b: op.b ?? -1, level: op.level, deg: op.deg };
    if (op.gal) o.gal = op.gal;
    o.k = op.k ?? (['smul', 'sadd', 'smac'].includes(op.op) ? 0 : -1);
    o.pt = op.pt ?? (['pmul', 'padd'].includes(op.op) ? 0 : -1);
    o.from = this.from;
    o.out = this.newReg(op.deg, op.op === 'rescale' ? op.level - 1 : op.level);
    this.ops.push(o);
    this.counts[op.op] = (this.counts[op.op] ?? 0) + 1;
    if (op.op === 'rotate') this.gal.add(op.gal);
    return o.out;
  }

  // A ciphertext: { reg (null for a fresh zero ciphertext not yet used), deg, level, scale }.
  input(level, scale) {
    return { reg: this.newReg(1, level), deg: 1, level, scale };
  }

  id(ct) {
    if (ct.reg === null) ct.reg = this.emit({ op: 'zero', level: ct.level, deg: ct.deg });
    return ct.reg;
  }

  zeroCt(deg, level, scale) {
    return { reg: null, deg, level, scale };
  }

  alias(ct) {
    this.id(ct);
    return { ...ct };
  }

  intConst(v, level) {
    return Uint32Array.from(this.p.q.slice(0, level + 1), (q) => bigMod(v, q));
  }

  compact(values, scale, level) {
    return encodeCompact(this.p, values, scale, level);
  }

  // smul of ct by an integer at level l (a new register; ct unchanged)
  smulInt(ct, k, l) {
    return { ...ct, reg: this.emit({ op: 'smul', a: this.id(ct), level: l, deg: ct.deg, k: this.words(this.intConst(k, l)) }), level: l };
  }

  // Add / Sub of two ciphertexts (Lattigo's evaluateInPlace): the operand with the smaller scale is first multiplied
  // by the integer part of the scale ratio (skipped when it is 1); the output has the larger scale.
  addCt(kind, a, b, outLevel) {
    const l = Math.min(a.level, b.level, outLevel);
    let ar = this.id(a);
    let br = this.id(b);
    const cmp = sCmp(a.scale, b.scale);
    if (cmp > 0) {
      const k = sDiv(a.scale, b.scale).toBigInt();
      if (k > 0n && k !== 1n) br = this.smulInt(b, k, l).reg;
    } else if (cmp < 0) {
      const k = sDiv(b.scale, a.scale).toBigInt();
      if (k > 0n && k !== 1n) ar = this.smulInt(a, k, l).reg;
    }
    const deg = Math.max(a.deg, b.deg);
    return { reg: this.emit({ op: kind, a: ar, b: br, level: l, deg }), deg, level: l, scale: sMax(a.scale, b.scale) };
  }

  // Add(op0, scalar): out.c0 += round(c * op0.scale)
  addScalar(a, c) {
    const K = this.intConst(roundScaled(toEncoding(c), a.scale), a.level);
    return { ...a, reg: this.emit({ op: 'sadd', a: this.id(a), level: a.level, deg: a.deg, k: this.words(K) }) };
  }

  // Add(op0, vector): the vector encoded at op0's scale
  addVector(a, v) {
    const pt = this.compact(v, a.scale, a.level);
    return { ...a, reg: this.emit({ op: 'padd', a: this.id(a), level: a.level, deg: a.deg, pt: this.words(pt) }) };
  }

  // Mul(op0, scalar): integers multiply as they are; anything else at scale q_level (the output scale grows by it)
  mulScalar(a, c) {
    const x = toEncoding(c);
    const l = a.level;
    const scale = x.isInt() ? scaleOf(1) : scaleOf(this.p.q[l]);
    const K = this.intConst(roundScaled(x, scale), l);
    return { reg: this.emit({ op: 'smul', a: this.id(a), level: l, deg: a.deg, k: this.words(K) }), deg: a.deg, level: l, scale: sMul(a.scale, scale) };
  }

  // Mul(op0, vector): the vector encoded at scale q_level
  mulVector(a, v) {
    const l = a.level;
    const ptScale = scaleOf(this.p.q[l]);
    const pt = this.compact(v, ptScale, l);
    return { reg: this.emit({ op: 'pmul', a: this.id(a), level: l, deg: a.deg, pt: this.words(pt) }), deg: a.deg, level: l, scale: sMul(a.scale, ptScale) };
  }

  // MulNew / MulRelinNew of two ciphertexts
  mulCt(a, b, relin) {
    const l = Math.min(a.level, b.level);
    let reg = this.emit({ op: 'tensor', a: this.id(a), b: this.id(b), level: l, deg: 2 });
    if (relin) reg = this.emit({ op: 'relin', a: reg, level: l, deg: 1 });
    return { reg, deg: relin ? 1 : 2, level: l, scale: sMul(a.scale, b.scale) };
  }

  relinearize(a) {
    return { ...a, reg: this.emit({ op: 'relin', a: this.id(a), level: a.level, deg: 1 }), deg: 1 };
  }

  rescale(a) {
    if (a.level <= 0) throw new Error(`The modulus chain is exhausted in ${this.from}`);
    const scale = sDiv(a.scale, scaleOf(this.p.q[a.level]));
    return { reg: this.emit({ op: 'rescale', a: this.id(a), level: a.level, deg: a.deg }), deg: a.deg, level: a.level - 1, scale };
  }

  rotate(a, k) {
    const gal = this.p.galois[this.p.rotations.indexOf(k)];
    if (gal === undefined) throw new Error(`The parameter set does not provide rotation ${k}`);
    return { reg: this.emit({ op: 'rotate', a: this.id(a), level: a.level, deg: 1, gal }), deg: 1, level: a.level, scale: a.scale };
  }

  // MulThenAdd(op0, scalar, acc): if the scales are equal and the scalar is not an integer, acc is first multiplied
  // by q_level; then acc += op0 * round(c * acc.scale / op0.scale).
  mulThenAdd(a, c, acc) {
    const x = toEncoding(c);
    const l = Math.min(a.level, acc.level);
    let accReg = this.id(acc);
    let accScale = acc.scale;
    let scaleRLWE;
    const cmp = sCmp(a.scale, acc.scale);
    if (cmp === 0) {
      if (x.isInt()) {
        scaleRLWE = scaleOf(1);
      } else {
        scaleRLWE = scaleOf(this.p.q[l]);
        accReg = this.emit({ op: 'smul', a: accReg, level: l, deg: acc.deg, k: this.words(this.intConst(BigInt(this.p.q[l]), l)) });
        accScale = sMul(accScale, scaleRLWE);
      }
    } else if (cmp < 0) {
      scaleRLWE = sDiv(acc.scale, a.scale);
    } else {
      throw new Error('MulThenAdd: op0.Scale > opOut.Scale is not supported');
    }
    const K = this.intConst(roundScaled(x, scaleRLWE), l);
    const deg = a.deg; // Lattigo resizes the receiver to op0's degree
    const reg = this.emit({ op: 'smac', a: this.id(a), b: accReg, level: l, deg, k: this.words(K) });
    return { reg, deg, level: acc.level, scale: accScale };
  }
}

// --- Chebyshev polynomials (bignum.Polynomial) and Lattigo's Paterson-Stockmeyer evaluation ---

function cloneC(c) {
  return { re: new BigFloat().set(c.re), im: new BigFloat().set(c.im) };
}

// Polynomial.Factorize(n) in the Chebyshev basis (with the circuit-level MaxDeg/Lead bookkeeping).
function factorize(p, n) {
  const deg = p.coeffs.length - 1;
  if (n < deg >> 1) throw new Error('cannot Factorize: n < p.Degree()/2');
  const pr = p.coeffs.slice(0, n).map(cloneC);
  const pq = new Array(deg - n + 1);
  pq[0] = cloneC(p.coeffs[n]);
  for (let i = n + 1, j = 1; i < deg + 1; i++, j++) {
    const c = cloneC(p.coeffs[i]);
    c.re.add(c.re, c.re);
    c.im.add(c.im, c.im);
    pq[i - n] = c;
    pr[n - j].re.sub(pr[n - j].re, p.coeffs[i].re);
    pr[n - j].im.sub(pr[n - j].im, p.coeffs[i].im);
  }
  const q = { coeffs: pq, maxDeg: p.maxDeg, lead: p.lead };
  const r = { coeffs: pr, maxDeg: p.maxDeg === deg ? n - 1 : p.maxDeg - (deg - n + 1), lead: false };
  return [q, r];
}

function optimalSplit(logDegree) {
  let logSplit = logDegree >> 1;
  const a = (1 << logSplit) + (1 << (logDegree - logSplit)) + logDegree - logSplit - 3;
  const b = (1 << (logSplit + 1)) + (1 << (logDegree - logSplit - 1)) + logDegree - logSplit - 4;
  if (a > b) logSplit++;
  return logSplit;
}

function splitDegree(n) {
  if ((n & (n - 1)) === 0) return [n / 2, n / 2];
  const k = bitsLen(n - 1) - 1;
  return [(1 << k) - 1, n + 1 - (1 << k)];
}

class PolyEvaluator {
  constructor(rec, p) {
    this.rec = rec;
    this.p = p;
  }

  // --- simulated levels and scales (the PS split's metadata) ---

  simGenPower(pb, n) {
    if (n < 2) return;
    const [a, b] = splitDegree(n);
    this.simGenPower(pb, a);
    this.simGenPower(pb, b);
    const level = Math.min(pb[a].level, pb[b].level);
    const scale = sMul(pb[a].scale, pb[b].scale);
    pb[n] = { level: level - 1, scale: sDiv(scale, scaleOf(this.p.q[level])) };
  }

  recursePS(logSplit, targetLevel, poly, pb, outputScale) {
    const deg = poly.coeffs.length - 1;
    if (deg < 1 << logSplit) {
      if (poly.lead && logSplit > 1 && poly.maxDeg > (1 << bitsLen(poly.maxDeg)) - (1 << (logSplit - 1))) {
        return this.recursePS(optimalSplit(bitsLen(deg)), targetLevel, poly, pb, outputScale);
      }
      const scale = poly.lead ? sMul(outputScale, scaleOf(this.p.q[targetLevel])) : outputScale;
      poly.level = targetLevel;
      poly.scale = scale;
      return [[poly], { level: targetLevel, scale }];
    }
    let nextPower = 1 << logSplit;
    while (nextPower < (deg >> 1) + 1) nextPower <<= 1;
    const xPow = pb[nextPower];
    const [cq, cr] = factorize(poly, nextPower);
    const qi = poly.lead ? this.p.q[targetLevel] : this.p.q[targetLevel + 1];
    const tScaleNew = sDiv(sMul(outputScale, scaleOf(qi)), xPow.scale);
    const [bsgsQ, res0] = this.recursePS(logSplit, targetLevel + 1, cq, pb, tScaleNew);
    const resc = { level: res0.level - 1, scale: sDiv(res0.scale, scaleOf(this.p.q[res0.level])) };
    const res = { level: Math.min(resc.level, xPow.level), scale: sMul(resc.scale, xPow.scale) };
    const [bsgsR] = this.recursePS(logSplit, targetLevel, cr, pb, res.scale);
    return [[...bsgsQ, ...bsgsR], res];
  }

  // --- the real power basis (common/polynomial.PowerBasis, Chebyshev) ---

  genPower(V, n) {
    if (!V[n]) {
      const rescale = this.genPowerInner(V, n);
      if (rescale) V[n] = this.rec.rescale(V[n]);
    }
  }

  genPowerInner(V, n) {
    if (V[n]) return false;
    const [a, b] = splitDegree(n);
    const rescaleA = this.genPowerInner(V, a);
    const rescaleB = this.genPowerInner(V, b);
    if (rescaleA) V[a] = this.rec.rescale(V[a]);
    if (rescaleB) V[b] = this.rec.rescale(V[b]);
    let x = this.rec.mulCt(V[a], V[b], true);
    const c = Math.abs(a - b);
    x = this.rec.addCt('add', x, x, x.level);
    if (c === 0) {
      x = this.rec.addScalar(x, -1);
    } else {
      V[n] = x; // GenPower(c) may read lower powers only; x stays the receiver
      this.genPower(V, c);
      x = this.rec.addCt('sub', x, V[c], x.level);
    }
    V[n] = x;
    return true;
  }

  // EvaluatePolynomialVectorFromPowerBasis for one polynomial (no slot mapping).
  evalFromPowerBasis(targetLevel, poly, V, targetScale) {
    const deg = poly.coeffs.length - 1;
    let maxCtDeg = 0;
    for (let i = deg; i > 0; i--) if (V[i]) maxCtDeg = Math.max(maxCtDeg, V[i].deg);
    let res = this.rec.zeroCt(deg === 0 ? 1 : maxCtDeg, targetLevel, targetScale);
    res = this.rec.addScalar(res, poly.coeffs[0].re);
    if (deg === 0) return res;
    for (let key = deg; key > 0; key--) res = this.rec.mulThenAdd(V[key], poly.coeffs[key].re, res);
    return res;
  }

  evaluate(x1, approx, targetScale) {
    const coeffs = approx.coeffs.map((c) => ({ re: c.re, im: c.im }));
    const poly = { coeffs, maxDeg: coeffs.length - 1, lead: true };
    const deg = coeffs.length - 1;
    const V = { 1: x1 };
    if (x1.level < bitsLen(deg) - 1) throw new Error('not enough levels for the polynomial');
    const logDegree = bitsLen(deg);
    const logSplit = optimalSplit(logDegree);
    this.genPower(V, 1 << (logDegree - 1));
    for (let i = (1 << logSplit) - 1; i > 2; i--) this.genPower(V, i);
    // The Paterson-Stockmeyer split, on simulated levels and scales.
    const pb = { 1: { level: x1.level, scale: x1.scale } };
    this.simGenPower(pb, 1 << logDegree);
    for (let i = (1 << logSplit) - 1; i > 2; i--) this.simGenPower(pb, i);
    const [ps] = this.recursePS(logSplit, x1.level - (bitsLen(deg) - 1), poly, pb, targetScale);
    // Baby steps (stored in reverse), then giant steps.
    const split = ps.length;
    let baby = new Array(split);
    for (let i = 0; i < split; i++) {
      const p = ps[i];
      baby[split - i - 1] = { degree: p.coeffs.length - 1, value: this.evalFromPowerBasis(p.level, p, V, p.scale) };
    }
    while (baby.length !== 1) {
      const giant = new Array(baby.length).fill(0);
      for (let i = 0; i < baby.length; i++) {
        if (i === baby.length - 1) giant[i] = 2;
        else if (baby[i].degree === baby[i + 1].degree) {
          giant[i] = 1;
          i++;
        }
      }
      for (let i = 0; i < baby.length; i++) {
        if (giant[i] === 2) {
          baby[i].degree = baby[i - 1].degree;
        } else if (giant[i] === 1) {
          const even = baby[i];
          const odd = baby[i + 1];
          const d = 1 << bitsLen(even.degree);
          // EvaluateMonomial(a = even, b = odd, X^d): b = rescale(relin(b)) * X^d + a
          let b = odd.value;
          if (b.deg === 2) b = this.rec.relinearize(b);
          b = this.rec.rescale(b);
          b = this.rec.mulCt(b, V[d], false);
          b = this.rec.addCt('add', b, even.value, b.level);
          odd.value = b;
          odd.degree = 2 * d - 1;
          baby[i] = null;
          i++;
        }
      }
      baby = baby.filter((s) => s !== null);
    }
    let out = baby[0].value;
    if (out.deg === 2) out = this.rec.relinearize(out);
    return this.rec.rescale(out);
  }
}

// --- the circuit (the removed evaluator.go, call for call) ---

function babyCount(n) {
  return n <= 16 ? 4 : 8;
}

function table(t, pos) {
  return Array.from({ length: DIM }, (_, i) => Array.from({ length: VOCAB }, (_, v) => t[pos][v][i]));
}

class Circuit {
  constructor(rec, pl, p) {
    this.rec = rec;
    this.pl = pl;
    this.p = p;
    this.poly = new PolyEvaluator(rec, p);
    this.keys = [];
    this.vals = [];
    this.mark = () => {};
  }

  cp(ct) {
    return this.rec.alias(ct);
  }

  rot(ct, k) {
    k = ((k % SLOTS) + SLOTS) % SLOTS;
    let out = ct;
    for (let bit = 0; bit < LOG_SLOTS; bit++) if (k & (1 << bit)) out = this.rec.rotate(out, 1 << bit);
    return out === ct ? this.cp(ct) : out;
  }

  rotsum(ct, n) {
    let acc = this.cp(ct);
    for (let i = 1; i < n; i *= 2) {
      const r = this.rot(acc, i);
      acc = this.rec.addCt('add', acc, r, acc.level);
    }
    return acc;
  }

  babies(x, b) {
    const out = [x];
    for (let i = 1; i < b; i++) out.push(this.rot(out[i - 1], 1));
    return out;
  }

  linear(bab, w, n, m) {
    const b = bab.length;
    let acc = null;
    for (let g = 0; g < n / b; g++) {
      let inner = null;
      for (let i = 0; i < b; i++) {
        const k = g * b + i;
        const d = new Array(SLOTS);
        for (let s = 0; s < SLOTS; s++) {
          const src = (((s - g * b) % SLOTS) + SLOTS) % SLOTS;
          d[s] = w[src % m][(src + k) % n];
        }
        const pr = this.rec.mulVector(bab[i], d);
        inner = inner === null ? pr : this.rec.addCt('add', inner, pr, inner.level);
      }
      inner = this.rec.rescale(inner);
      const r = this.rot(inner, g * b);
      acc = acc === null ? r : this.rec.addCt('add', acc, r, acc.level);
    }
    return acc;
  }

  mulRelin(a, b) {
    return this.rec.rescale(this.rec.mulCt(a, b, true));
  }

  approx(ct, a) {
    return this.poly.evaluate(this.cp(ct), a, scaleOf(2 ** this.p.logScale));
  }

  lookup(onehot, pos, last) {
    const bab = this.babies(onehot, babyCount(VOCAB));
    const k = this.linear(bab, table(this.pl.k, pos), VOCAB, DIM);
    const v = this.linear(bab, table(this.pl.v, pos), VOCAB, DIM);
    this.keys.push(k);
    this.vals.push(v);
    if (!last) return [null, null];
    const q = this.linear(bab, table(this.pl.q, pos), VOCAB, DIM);
    const x0 = this.linear(bab, table(this.pl.x0, pos), VOCAB, DIM);
    return [q, x0];
  }

  layerNorm(h, isq) {
    const s = this.rotsum(h, DIM);
    let cc = this.rec.mulScalar(h, DIM);
    cc = this.rec.addCt('sub', cc, s, cc.level);
    const sq = this.mulRelin(cc, cc);
    let v = this.rotsum(sq, DIM);
    v = this.rec.addScalar(v, this.pl.eps * DIM * DIM * DIM);
    let t = this.rec.mulScalar(v, isq.scalar);
    t = this.rec.rescale(t);
    t = this.rec.addScalar(t, isq.offset);
    const r = this.approx(t, isq);
    return this.mulRelin(cc, r);
  }

  forward(q, x0) {
    const pl = this.pl;
    const es = [];
    let sum = null;
    for (const key of this.keys) {
      const pr = this.mulRelin(q, key);
      let s = this.rotsum(pr, DIM);
      s = this.rec.addScalar(s, pl.expP.offset);
      const e = this.approx(s, pl.expP);
      es.push(e);
      sum = sum === null ? this.cp(e) : this.rec.addCt('add', sum, e, sum.level);
    }
    this.mark('softmax');
    sum = this.rec.addScalar(sum, pl.invP.offset);
    const inv = this.approx(sum, pl.invP);
    this.mark('attention-values');
    let h = null;
    es.forEach((e, j) => {
      const a = this.mulRelin(e, inv);
      const o = this.mulRelin(a, this.vals[j]);
      h = h === null ? o : this.rec.addCt('add', h, o, h.level);
    });
    h = this.rec.addCt('add', h, x0, h.level);
    h = this.rec.addVector(h, replicate(pl.projB, DIM));
    this.mark('ln2');
    const y = this.layerNorm(h, pl.isq2P);
    this.mark('mlp');
    let bab = this.babies(y, babyCount(DIM));
    let u = this.linear(bab, pl.fc1W, DIM, HIDDEN);
    u = this.rec.addVector(u, replicate(pl.fc1B, HIDDEN));
    const g = this.approx(u, pl.geluP);
    bab = this.babies(g, babyCount(HIDDEN));
    let mlp = this.linear(bab, pl.fc2W, HIDDEN, DIM);
    mlp = this.rec.addVector(mlp, replicate(pl.fc2B, DIM));
    mlp = this.rec.addCt('add', mlp, h, mlp.level);
    this.mark('lnf');
    const yf = this.layerNorm(mlp, pl.isqfP);
    this.mark('head');
    bab = this.babies(yf, babyCount(DIM));
    const logits = this.linear(bab, pl.headW, DIM, VOCAB);
    return this.rec.addVector(logits, replicate(pl.headB, VOCAB));
  }
}

class TriageCircuit extends Circuit {
  constructor(rec, model, profile) {
    super(rec, model, profile);
    this.values = Array.from({ length: model.width }, (_, i) => model.values.map(row => row[i]));
    this.weightedValues = this.values.map(row => row.map((value, token) => value * model.keys[token]));
  }

  lookup(onehot) {
    const bab = this.babies(onehot, babyCount(VOCAB));
    this.vals.push(this.linear(bab, this.values, VOCAB, this.pl.width));
    // The key/value product is a public embedding lookup. Store both encrypted moments, so a new query can
    // combine prior positions without re-uploading their inputs or depleting their levels.
    this.keys.push(this.linear(bab, this.weightedValues, VOCAB, this.pl.width));
    return [null, null];
  }

  forward() {
    const { width, hidden, head } = this.pl;
    const count = this.vals.length;
    const combine = (a, b) => this.rec.addCt('add', a, b, a.level);
    const weighted = this.keys.reduce(combine);
    const pooled = this.rec.addCt(queryAt(count) === 1 ? 'add' : 'sub', this.vals.reduce(combine), weighted, weighted.level);
    this.mark('mlp');
    const block = Math.min(width, babyCount(width));
    const u = this.rec.addVector(this.linear(this.babies(pooled, block), hidden.weight.map(row => row.map(x => x / count)),
      width, width), replicate(hidden.bias, width));
    const factor = this.rec.addScalar(this.rec.rescale(this.rec.mulScalar(u, QUADRATIC_TERM)), 0.5);
    const activated = this.mulRelin(u, factor);
    this.mark('head');
    const logits = this.linear(this.babies(activated, block), head.weight, width, VOCAB);
    return this.rec.addVector(logits, replicate(head.bias, VOCAB));
  }
}

// --- steps, state and register lifetimes ---

function finishSteps(prog, names) {
  const def = new Map();
  prog.steps.forEach((st, si) => {
    for (const r of st.inputs) def.set(r, si);
    for (let i = st.opFirst; i < st.opEnd; i++) def.set(prog.ops[i].out, si);
  });
  const lastStep = new Map();
  const lastUse = new Map();
  prog.steps.forEach((st, si) => {
    for (let i = st.opFirst; i < st.opEnd; i++) {
      for (const r of [prog.ops[i].a, prog.ops[i].b]) {
        if (r < 0) continue;
        lastStep.set(r, si);
        if (def.get(r) === si) lastUse.set(r, i);
      }
    }
  });
  prog.steps.forEach((st, si) => {
    const seenIn = new Set();
    st.stateIn = [];
    st.stateOut = [];
    for (let i = st.opFirst; i < st.opEnd; i++) {
      for (const r of [prog.ops[i].a, prog.ops[i].b]) {
        if (r < 0 || def.get(r) === si || seenIn.has(r)) continue;
        if (!names.has(r)) throw new Error(`step ${si} reads register ${r} from an earlier step that is not a KV cache entry`);
        seenIn.add(r);
        st.stateIn.push({ name: names.get(r), reg: r });
      }
    }
    for (const [reg, d] of def) {
      if (d !== si) continue;
      if (names.has(reg)) st.stateOut.push({ name: names.get(reg), reg });
      else if ((lastStep.get(reg) ?? -1) > si) throw new Error(`register ${reg} of step ${si} is used later but is not a KV cache entry`);
    }
    st.stateIn.sort((x, y) => x.reg - y.reg);
    st.stateOut.sort((x, y) => x.reg - y.reg);
    st.counts = {};
    for (let i = st.opFirst; i < st.opEnd; i++) st.counts[prog.ops[i].op] = (st.counts[prog.ops[i].op] ?? 0) + 1;
  });
  for (const [reg, d] of def) {
    if ((lastStep.get(reg) ?? -1) > d || reg === prog.steps[d].output || names.has(reg)) continue;
    let at = lastUse.get(reg);
    if (at === undefined) {
      for (let i = prog.steps[d].opFirst; i < prog.steps[d].opEnd; i++) if (prog.ops[i].out === reg) at = i;
    }
    (prog.ops[at].free ??= []).push(reg);
  }
  for (const op of prog.ops) op.free?.sort((x, y) => x - y);
}

// --- JSON in the Go encoder's field order (maps with sorted keys, nil slices as null) ---

function sortedMap(m) {
  const out = {};
  for (const k of Object.keys(m).sort()) out[k] = m[k];
  return out;
}

export function programJSON(prog) {
  const ops = prog.ops.map((o) => {
    const x = { op: o.op, out: o.out, a: o.a, b: o.b, level: o.level, deg: o.deg };
    if (o.gal) x.gal = o.gal;
    x.k = o.k;
    x.pt = o.pt;
    if (o.free?.length) x.free = o.free;
    x.from = o.from;
    return x;
  });
  const steps = prog.steps.map((s) => ({
    kind: s.kind, first: s.first, position: s.position, opFirst: s.opFirst, opEnd: s.opEnd, inputs: s.inputs,
    stateIn: s.stateIn.length ? s.stateIn : null, stateOut: s.stateOut.length ? s.stateOut : null, output: s.output,
    outputLevel: s.outputLevel, outputScale: s.outputScale, layers: s.layers, counts: sortedMap(s.counts),
  }));
  return JSON.stringify({
    schema: prog.schema, protocol: prog.protocol, profile: prog.profile, profileDigest: prog.profileDigest, model: prog.model,
    ...(prog.circuit ? { circuit: prog.circuit, rotations: prog.rotations } : {}),
    logN: prog.logN, logSlots: prog.logSlots, q: prog.q, p: prog.p, psi: prog.psi, galois: prog.galois, inputScale: prog.inputScale,
    regs: prog.regs, ops, steps, blobWords: prog.blobWords, blobDigest: prog.blobDigest, counts: sortedMap(prog.counts),
  });
}

// compileProgram: the prefill of `prompt` tokens, then one decode step per position up to and including
// decodeThrough (none when decodeThrough < prompt). Returns { program, blob (Uint32Array), json }.
export function compileProgram({ profile: profileId, modelPath, prompt, decodeThrough, triage = false, log = () => {} }) {
  const p = lookupProfile(profileId);
  if (p.wire !== 'raw-u32/v0') throw new Error(`profile ${p.id} is not a u32 (WebGPU) profile`);
  const context = triage ? TRIAGE_CONTEXT : CONTEXT;
  if (!Number.isInteger(prompt) || !Number.isInteger(decodeThrough) || prompt < 1 || prompt > context || decodeThrough >= context) {
    throw new Error(`prompt ${prompt} / decode-through ${decodeThrough} outside the ${context}-position context`);
  }
  const { model, id: modelId, circuit } = triage ? loadTriageModel(modelPath) : loadModel(modelPath);
  const rec = new Recorder(p);
  const cir = triage ? new TriageCircuit(rec, model, p) : new Circuit(rec, newPlan(model), p);
  const inputScale = scaleOf(2 ** p.logScale);
  const names = new Map();
  const steps = [];
  let layers = [];
  let first = 0;
  const mark = (name) => {
    if (rec.ops.length > first) layers.push({ name: rec.from, first, last: rec.ops.length - 1 });
    first = rec.ops.length;
    rec.from = name;
  };
  cir.mark = mark;
  const t0 = performance.now();
  const record = (kind, from, to) => {
    const st = { kind, first: from, position: to, opFirst: rec.ops.length, inputs: [] };
    layers = [];
    let q = null;
    let x0 = null;
    for (let pos = from; pos <= to; pos++) {
      const ct = rec.input(p.maxLevel, inputScale);
      st.inputs.push(ct.reg);
      mark(`embed-${pos}`);
      [q, x0] = cir.lookup(ct, pos, pos === to);
      names.set(rec.id(cir.keys[pos]), `k${pos}`);
      names.set(rec.id(cir.vals[pos]), `v${pos}`);
    }
    mark('attention-scores');
    const logits = cir.forward(q, x0);
    mark('end');
    st.output = rec.id(logits);
    st.outputLevel = logits.level;
    st.outputScale = logits.scale.toNumber();
    st.opEnd = rec.ops.length;
    st.layers = layers;
    steps.push(st);
    log(`recorded ${kind} at position ${to}: ops ${st.opFirst}..${st.opEnd - 1} (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  };
  record('prefill', 0, prompt - 1);
  for (let pos = prompt; pos <= decodeThrough; pos++) record('decode', pos, pos);
  const prog = {
    schema: PROGRAM_SCHEMA, protocol: PROTOCOL_ID, profile: p.id, profileDigest: p.digest, model: modelId, circuit, logN: p.logN,
    logSlots: LOG_SLOTS, q: p.q, p: p.p, psi: p.psi, rotations: p.rotations, galois: [...rec.gal].sort((x, y) => x - y), inputScale: inputScale.toNumber(),
    regs: rec.regs, ops: rec.ops, steps, counts: rec.counts,
  };
  finishSteps(prog, names);
  const blob = new Uint32Array(rec.blobLen);
  let at = 0;
  for (const part of rec.blob) {
    blob.set(part, at);
    at += part.length;
  }
  prog.blobWords = blob.length;
  prog.blobDigest = `sha256:${createHash('sha256').update(Buffer.from(blob.buffer)).digest('hex')}`;
  const json = programJSON(prog);
  log(`session: ${steps.length} steps, ${prog.ops.length} ops, ${prog.regs.length} registers, ${prog.blobWords} constant words (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  return { program: JSON.parse(json), blob, json };
}
