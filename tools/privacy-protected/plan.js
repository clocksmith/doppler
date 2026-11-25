// The public model plan of the tiny protected fixture: weights folded into lookup tables, matrices and Chebyshev
// approximations, and the plaintext references (the float64 model and the approximated circuit). The requester never
// loads this module's inputs; the compiler, the trap oracle and the post-route comparison do.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { BigFloat, Complex, ComplexMultiplier, cos, newFloat, pi } from './ckks/bigfloat.js';
import { goExp, goGeluTanh } from './ckks/gomath.js';

export const PROTOCOL_ID = 'doppler-protected-ckks-tiny/v0-research';
export const VOCAB = 32;
export const CONTEXT = 8;
export const DIM = 16;
export const HIDDEN = 64;
export const LOG_SLOTS = 6;
export const SLOTS = 1 << LOG_SLOTS;
export const ALPHABET = "abcdefghijklmnopqrstuvwxyz .,!?'";
export const EXP_SHIFT = 8.0;
export const MODEL_SCHEMA = 'doppler.protected-tiny-model/v1';
export const DEFAULT_MODEL = new URL('./tiny-model.json', import.meta.url);

export function tokenize(s) {
  const out = [];
  for (const ch of s) {
    const i = ALPHABET.indexOf(ch);
    if (i < 0) throw new Error(`symbol ${JSON.stringify(ch)} is outside the tokenizer alphabet`);
    out.push(i);
  }
  return out;
}

export function detokenize(toks) {
  return toks.map((t) => ALPHABET[t]).join('');
}

// loadModel: the fixture and its identity (sha256 of the exact file bytes).
export function loadModel(path = DEFAULT_MODEL) {
  const raw = readFileSync(path);
  const m = JSON.parse(raw.toString('utf8'));
  if (m.schema !== MODEL_SCHEMA || m.vocab !== VOCAB || m.context !== CONTEXT || m.dim !== DIM || m.hidden !== HIDDEN) {
    throw new Error('model geometry does not match the compiled plan');
  }
  const w = m.weights;
  if (w.wte?.length !== VOCAB || w.wpe?.length !== CONTEXT || w.ln1_g?.length !== DIM || w.attn_w?.length !== 3 * DIM
    || w.proj_w?.length !== DIM || w.ln2_g?.length !== DIM || w.fc1_w?.length !== HIDDEN || w.fc2_w?.length !== DIM
    || w.lnf_g?.length !== DIM || !(w.ln_eps > 0)) {
    throw new Error('model weights are incomplete');
  }
  return { model: m, id: `sha256:${createHash('sha256').update(raw).digest('hex')}` };
}

// --- Chebyshev approximation (Lattigo's bignum.ChebyshevApproximation, at 128-bit precision) ---

function chebyshevNodes(n, A, B) {
  const prec = A.prec;
  const nodes = new Array(n);
  const half = new BigFloat(prec).setFloat64(0.5);
  const x = new BigFloat().add(A, B);
  x.mul(x, half);
  const y = new BigFloat().sub(B, A);
  y.mul(y, half);
  const piOverN = pi(prec);
  piOverN.quo(piOverN, new BigFloat().setInt64(n));
  for (let k = 1; k < n + 1; k++) {
    let up = new BigFloat(prec).setFloat64(k - 0.5);
    up.mul(up, piOverN);
    up = cos(up);
    up.mul(up, y);
    up.add(up, x);
    nodes[n - k] = up;
  }
  return nodes;
}

function chebyCoeffs(nodes, fi, A, B) {
  const prec = A.prec;
  const n = nodes.length;
  const coeffs = Array.from({ length: n }, () => Complex.withPrec(prec));
  const u = Complex.withPrec(prec);
  const mul = new ComplexMultiplier();
  const tmp = Complex.withPrec(prec);
  const two = new BigFloat(prec).setInt64(2);
  const minusab = new BigFloat().set(A);
  minusab.neg_(minusab);
  minusab.sub(minusab, B);
  const bminusa = new BigFloat().set(B);
  bminusa.sub(bminusa, A);
  const Tnext = Complex.withPrec(prec);
  for (let i = 0; i < n; i++) {
    u.re.mul(nodes[i], two);
    u.re.add(u.re, minusab);
    u.re.quo(u.re, bminusa);
    const Tprev = Complex.withPrec(prec);
    Tprev.re.setFloat64(1);
    const T = u.clone();
    for (let j = 0; j < n; j++) {
      mul.mul(fi[i], Tprev, tmp);
      coeffs[j].add(coeffs[j], tmp);
      mul.mul(u, T, Tnext);
      Tnext.re.mul(Tnext.re, two);
      Tnext.im.mul(Tnext.im, two);
      Tnext.sub(Tnext, Tprev);
      Tprev.set(T);
      T.set(Tnext);
    }
  }
  const nHalf = new BigFloat().setInt64(n);
  coeffs[0].re.quo(coeffs[0].re, nHalf);
  coeffs[0].im.quo(coeffs[0].im, nHalf);
  nHalf.quo(nHalf, two);
  for (let i = 1; i < n; i++) {
    coeffs[i].re.quo(coeffs[i].re, nHalf);
    coeffs[i].im.quo(coeffs[i].im, nHalf);
  }
  return coeffs;
}

// An approximation: the Chebyshev interpolant of f on [a, b] with `degree + 1` nodes. Inputs reach the encrypted
// polynomial already mapped to [-1, 1] by t = scalar * x + offset.
export function newApproximation(name, a, b, degree, f) {
  const A = newFloat(a, 128);
  const B = newFloat(b, 128);
  const nodes = chebyshevNodes(degree + 1, A, B);
  const fi = nodes.map((node) => {
    const x = new BigFloat(128).set(node);
    return new Complex(new BigFloat().setFloat64(f(x.toNumber())), new BigFloat());
  });
  const coeffs = chebyCoeffs(nodes, fi, A, B);
  return {
    name, a, b, degree, A, B, coeffs,
    c: coeffs.map((c) => c.re.toNumber()),
    scalar: 2 / (b - a),
    offset: -(a + b) / (b - a),
  };
}

// Clenshaw evaluation at an already mapped t, in float64.
export function evalT(p, t) {
  let b1 = 0;
  let b2 = 0;
  for (let k = p.c.length - 1; k >= 1; k--) {
    const nb1 = 2 * t * b1 - b2 + p.c[k];
    b2 = b1;
    b1 = nb1;
  }
  return t * b1 - b2 + p.c[0];
}

export function evalX(p, x) {
  return evalT(p, p.scalar * x + p.offset);
}

export function geluTanh(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

function widen(r, lo, hi) {
  return [Math.min(r[0], lo), Math.max(r[1], hi)];
}

export function layerNorm(x, g, b, eps) {
  let mu = 0;
  for (const a of x) mu += a;
  mu /= x.length;
  let v = 0;
  for (const a of x) v += (a - mu) * (a - mu);
  v /= x.length;
  return x.map((a, i) => ((a - mu) / Math.sqrt(v + eps)) * g[i] + b[i]);
}

export function matVec(w, x) {
  return w.map((row) => {
    let s = 0;
    for (let j = 0; j < x.length; j++) s += row[j] * x[j];
    return s;
  });
}

export function affine(w, b, x) {
  return matVec(w, x).map((v, i) => v + b[i]);
}

// newPlan: everything the evaluator computes with (the removed plan.go). The approximated functions are evaluated at the Chebyshev
// nodes with Go's float64 exp and tanh (ckks/gomath.js), so the coefficients equal the ones the Go tools derived.
export function newPlan(m) {
  const w = m.weights;
  const pl = { m, eps: w.ln_eps };
  const [sLo, sHi] = widen(m.calibration.scores, -8, 9);
  pl.invP = newApproximation('inverse', 0.08, 4.0, 15, (x) => 1 / x);
  const alphaInv = pl.invP.scalar;
  pl.expP = newApproximation('exp', sLo, sHi, 15, (x) => alphaInv * goExp(x - EXP_SHIFT));
  const d3 = DIM ** 3;
  pl.isq2P = newApproximation('ln2-inverse-sqrt', d3 * (0.12 + w.ln_eps), d3 * (2.0 + w.ln_eps), 15, (x) => 1 / Math.sqrt(x));
  pl.isqfP = newApproximation('lnf-inverse-sqrt', d3 * (0.04 + w.ln_eps), d3 * (13 + w.ln_eps), 31, (x) => 1 / Math.sqrt(x));
  const [gLo, gHi] = widen(m.calibration.gelu_in, -13, 8);
  pl.geluP = newApproximation('gelu', gLo, gHi, 31, goGeluTanh);

  const expScale = pl.expP.scalar;
  const sqrtD = Math.sqrt(DIM);
  pl.x0 = [];
  pl.q = [];
  pl.k = [];
  pl.v = [];
  for (let p = 0; p < CONTEXT; p++) {
    pl.x0.push([]);
    pl.q.push([]);
    pl.k.push([]);
    pl.v.push([]);
    for (let tok = 0; tok < VOCAB; tok++) {
      const x = w.wte[tok].map((v, j) => v + w.wpe[p][j]);
      const a = layerNorm(x, w.ln1_g, w.ln1_b, w.ln_eps);
      const qkv = affine(w.attn_w, w.attn_b, a);
      const q = qkv.slice(0, DIM);
      const k = qkv.slice(DIM, 2 * DIM);
      const v = qkv.slice(2 * DIM);
      const qs = q.map((x) => (x * expScale) / sqrtD);
      const vo = matVec(w.proj_w, v).map((x) => x / alphaInv);
      pl.x0[p].push(x);
      pl.q[p].push(qs);
      pl.k[p].push(k);
      pl.v[p].push(vo);
    }
  }
  pl.projB = w.proj_b;
  const ga = pl.geluP.scalar;
  const gb = pl.geluP.offset;
  pl.fc1W = [];
  pl.fc1B = [];
  for (let i = 0; i < HIDDEN; i++) {
    const row = [];
    let bias = w.fc1_b[i];
    for (let j = 0; j < DIM; j++) {
      row.push(ga * w.fc1_w[i][j] * sqrtD * w.ln2_g[j]);
      bias += w.fc1_w[i][j] * w.ln2_b[j];
    }
    pl.fc1W.push(row);
    pl.fc1B.push(ga * bias + gb);
  }
  pl.fc2W = w.fc2_w;
  pl.fc2B = w.fc2_b;
  pl.headW = [];
  pl.headB = [];
  for (let t = 0; t < VOCAB; t++) {
    const row = [];
    let b = 0;
    for (let j = 0; j < DIM; j++) {
      row.push(w.wte[t][j] * sqrtD * w.lnf_g[j]);
      b += w.wte[t][j] * w.lnf_b[j];
    }
    pl.headW.push(row);
    pl.headB.push(b);
  }
  return pl;
}

// Every float64 of the plan tables in a fixed order (for the golden digest).
export function planTables(pl) {
  const rows = [];
  for (const t of [pl.x0, pl.q, pl.k, pl.v]) for (let p = 0; p < CONTEXT; p++) for (let v = 0; v < VOCAB; v++) rows.push(t[p][v]);
  rows.push(pl.projB, ...pl.fc1W, pl.fc1B, ...pl.fc2W, pl.fc2B, ...pl.headW, pl.headB);
  return rows;
}

// --- plaintext references ---

export function exactLogits(m, toks) {
  const w = m.weights;
  const t = toks.length;
  const xs = [];
  const ks = [];
  const vs = [];
  let q = null;
  toks.forEach((tok, p) => {
    const x = w.wte[tok].map((v, j) => v + w.wpe[p][j]);
    const qkv = affine(w.attn_w, w.attn_b, layerNorm(x, w.ln1_g, w.ln1_b, w.ln_eps));
    xs.push(x);
    q = qkv.slice(0, DIM);
    ks.push(qkv.slice(DIM, 2 * DIM));
    vs.push(qkv.slice(2 * DIM));
  });
  const s = new Array(t).fill(0);
  let mx = -Infinity;
  for (let j = 0; j < t; j++) {
    for (let i = 0; i < DIM; i++) s[j] += q[i] * ks[j][i];
    s[j] /= Math.sqrt(DIM);
    mx = Math.max(mx, s[j]);
  }
  let z = 0;
  for (let j = 0; j < t; j++) {
    s[j] = Math.exp(s[j] - mx);
    z += s[j];
  }
  const o = new Array(DIM).fill(0);
  for (let j = 0; j < t; j++) for (let i = 0; i < DIM; i++) o[i] += (s[j] / z) * vs[j][i];
  const h = affine(w.proj_w, w.proj_b, o).map((v, i) => v + xs[t - 1][i]);
  const u = affine(w.fc1_w, w.fc1_b, layerNorm(h, w.ln2_g, w.ln2_b, w.ln_eps)).map(geluTanh);
  const mlp = affine(w.fc2_w, w.fc2_b, u);
  const h2 = h.map((v, i) => v + mlp[i]);
  return matVec(w.wte, layerNorm(h2, w.lnf_g, w.lnf_b, w.ln_eps));
}

// trace: counts polynomial inputs outside their interval (range audits).
export function newTrace() {
  return { outOfRange: {} };
}

function check(tr, p, x) {
  if (tr && (x < p.a || x > p.b)) tr.outOfRange[p.name] = (tr.outOfRange[p.name] ?? 0) + 1;
}

// lnFolded mirrors the encrypted layer norm: c' = D*h - sum(h), var' = sum(c'^2), y = c' * isq(var' + eps*D^3).
function lnFolded(h, isq, eps, tr) {
  let s = 0;
  for (const a of h) s += a;
  let v = 0;
  const c = h.map((a) => {
    const x = DIM * a - s;
    v += x * x;
    return x;
  });
  const x = v + eps * DIM ** 3;
  check(tr, isq, x);
  const r = evalX(isq, x);
  return c.map((a) => a * r);
}

export function approxLogits(pl, toks, tr = null) {
  const t = toks.length;
  const pos = t - 1;
  const tok = toks[t - 1];
  const q = pl.q[pos][tok];
  const es = [];
  let sum = 0;
  for (let j = 0; j < t; j++) {
    let tt = 0;
    for (let i = 0; i < DIM; i++) tt += q[i] * pl.k[j][toks[j]][i];
    tt += pl.expP.offset;
    check(tr, pl.expP, (tt - pl.expP.offset) / pl.expP.scalar);
    es.push(evalT(pl.expP, tt));
    sum += es[j];
  }
  const tInv = sum + pl.invP.offset;
  check(tr, pl.invP, (tInv - pl.invP.offset) / pl.invP.scalar);
  const inv = evalT(pl.invP, tInv);
  const h = new Array(DIM);
  for (let i = 0; i < DIM; i++) {
    h[i] = pl.x0[pos][tok][i] + pl.projB[i];
    for (let j = 0; j < t; j++) h[i] += es[j] * inv * pl.v[j][toks[j]][i];
  }
  const y = lnFolded(h, pl.isq2P, pl.eps, tr);
  const u = affine(pl.fc1W, pl.fc1B, y).map((x) => {
    check(tr, pl.geluP, (x - pl.geluP.offset) / pl.geluP.scalar);
    return evalT(pl.geluP, x);
  });
  const mlp = affine(pl.fc2W, pl.fc2B, u);
  const h2 = h.map((v, i) => v + mlp[i]);
  return affine(pl.headW, pl.headB, lnFolded(h2, pl.isqfP, pl.eps, tr));
}

export function argmax(x) {
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

export function margin(x) {
  const b = argmax(x);
  let second = -Infinity;
  x.forEach((a, i) => {
    if (i !== b && a > second) second = a;
  });
  return x[b] - second;
}

export function greedy(prompt, steps, logits) {
  const toks = [...prompt];
  const all = [];
  for (let i = 0; i < steps; i++) {
    const l = logits(toks);
    all.push(l);
    toks.push(argmax(l));
  }
  return { tokens: toks.slice(prompt.length), logits: all };
}

// Go's %v for a float64 (shortest digits; exponent form when the decimal exponent is below -4 or at least 6).
export function goFloatV(f) {
  if (f === 0) return Object.is(f, -0) ? '-0' : '0';
  const s = Math.abs(f).toExponential(); // shortest digits, d.ddde±x
  const [mant, expS] = s.split('e');
  const exp = Number(expS);
  const digits = mant.replace('.', '');
  const sign = f < 0 ? '-' : '';
  if (exp < -4 || exp >= 6) {
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    const ae = Math.abs(exp);
    return `${sign}${m}e${exp < 0 ? '-' : '+'}${ae < 10 ? `0${ae}` : ae}`;
  }
  if (exp < 0) return `${sign}0.${'0'.repeat(-exp - 1)}${digits}`;
  if (digits.length <= exp + 1) return `${sign}${digits}${'0'.repeat(exp + 1 - digits.length)}`;
  return `${sign}${digits.slice(0, exp + 1)}.${digits.slice(exp + 1)}`;
}

// circuitDigest identifies the exact approximate circuit: protocol, model and every approximation (interval, degree).
// Known-answer traps are generated for, and bound to, this digest.
export function circuitDigest(pl, modelId) {
  let s = `doppler.protected-circuit/v1|${PROTOCOL_ID}|${modelId}|${VOCAB}|${CONTEXT}|${DIM}|${HIDDEN}|${goFloatV(EXP_SHIFT)}|${goFloatV(pl.eps)}`;
  for (const p of [pl.expP, pl.invP, pl.isq2P, pl.isqfP, pl.geluP]) s += `|${p.name}:${goFloatV(p.a)}:${goFloatV(p.b)}:${p.degree}`;
  return `sha256:${createHash('sha256').update(s).digest('hex')}`;
}

export function approximations(pl) {
  return [pl.expP, pl.invP, pl.isq2P, pl.isqfP, pl.geluP];
}

export function replicate(v, period) {
  const out = new Array(SLOTS).fill(0);
  for (let s = 0; s < SLOTS; s++) if (s % period < v.length) out[s] = v[s % period];
  return out;
}
