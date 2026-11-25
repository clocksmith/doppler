// The requester's secret-key side of single-key RLWE/CKKS: secret key, evaluation keys, encryption and decryption.
// All randomness comes from the platform CSPRNG (node:crypto randomFillSync). Nothing in this module is imported by
// the evaluator (webgpu/) or by the compiler; the evaluator never holds a secret key.
//
// Distributions follow Lattigo's defaults: dense ternary secret (each coefficient 0 with probability 1/3, +1 or -1
// with probability 1/3 each) and a discrete Gaussian error with sigma 3.2 truncated at 19.2 (|e| <= 19).
// Evaluation keys use the hybrid RNS gadget (digit i = Q primes [i*|P|, (i+1)*|P|)) in canonical form, as the raw
// key wire format and the WebGPU key switch expect.

import { randomFillSync } from 'node:crypto';
import { LOG_SLOTS } from '../plan.js';
import { ROTATIONS, ringOf } from './params.js';
import { COMPACT_WORDS, decodeCoeffs, encodeCompact, expandCompact } from './encoder.js';
import { automorphismIndex, mulmod, mulmodInv, modInvGaloisElement } from './ring.js';
import { RAW_CT_HEADER, RAW_CT_MAGIC, RAW_KEY_HEADER, RAW_KEY_MAGIC, decodeCt, encodeCt } from './codec.js';
import { BigFloat } from './bigfloat.js';

export const SIGMA = 3.2;
export const BOUND = 19;

// --- CSPRNG streams ---

class RandomStream {
  constructor(bytes = 1 << 22) {
    this.buf = new Uint32Array(bytes >> 2);
    this.at = this.buf.length;
  }

  u32() {
    if (this.at === this.buf.length) {
      randomFillSync(this.buf);
      this.at = 0;
    }
    return this.buf[this.at++];
  }

  // Uniform in [0, q) by rejection (q < 2^32).
  below(q) {
    const lim = Math.floor(2 ** 32 / q) * q;
    for (;;) {
      const r = this.u32();
      if (r < lim) return r % q;
    }
  }

  // Uniform float in [0, 1) with 53 random bits.
  unit() {
    return (this.u32() * 2 ** 21 + (this.u32() >>> 11)) / 2 ** 53;
  }
}

// Discrete Gaussian on [-BOUND, BOUND] by inverse CDF over exact-enough float64 weights.
const GAUSS_CDF = (() => {
  const w = [];
  let total = 0;
  for (let x = -BOUND; x <= BOUND; x++) {
    const v = Math.exp(-(x * x) / (2 * SIGMA * SIGMA));
    w.push(v);
    total += v;
  }
  let acc = 0;
  return w.map((v) => (acc += v / total));
})();

function sampleGaussian(rng, n) {
  const out = new Int8Array(n);
  for (let j = 0; j < n; j++) {
    const u = rng.unit();
    let i = 0;
    while (i < GAUSS_CDF.length - 1 && u >= GAUSS_CDF[i]) i++;
    out[j] = i - BOUND;
  }
  return out;
}

function sampleTernary(rng, n) {
  const out = new Int8Array(n);
  for (let j = 0; j < n; j++) {
    let r;
    do r = rng.u32() & 3; while (r === 3);
    out[j] = r - 1;
  }
  return out;
}

// Small signed integers into the NTT domain of one modulus.
function smallNTT(mod, small) {
  const q = mod.q;
  const row = new Float64Array(mod.n);
  for (let j = 0; j < mod.n; j++) row[j] = small[j] < 0 ? q + small[j] : small[j];
  return mod.ntt(row);
}

// --- secret key ---

export class SecretKey {
  constructor(profile, coeffs) {
    if (coeffs.length !== profile.n || coeffs.some((c) => c < -1 || c > 1)) throw new Error('secret key is not ternary of length N');
    this.p = profile;
    this.coeffs = coeffs;
    this.ntt = new Map(); // global modulus index -> Float64Array
  }

  static generate(profile, rng = new RandomStream()) {
    return new SecretKey(profile, sampleTernary(rng, profile.n));
  }

  // limb g: 0..qCount-1 are Q, qCount.. are P.
  at(g) {
    let v = this.ntt.get(g);
    if (!v) {
      const ring = ringOf(this.p);
      const mod = g < this.p.qCount ? ring.Q[g] : ring.P[g - this.p.qCount];
      v = smallNTT(mod, this.coeffs);
      this.ntt.set(g, v);
    }
    return v;
  }

  // Serialized secret: magic, profile digest, ternary coefficients as bytes (0, 1, 2 = -1, 0, +1 shifted by 1).
  toBytes() {
    const id = Buffer.from(this.p.digest);
    const out = Buffer.alloc(8 + id.length + this.p.n);
    out.write('DHS0', 0);
    out.writeUInt32LE(id.length, 4);
    id.copy(out, 8);
    for (let j = 0; j < this.p.n; j++) out[8 + id.length + j] = this.coeffs[j] + 1;
    return out;
  }

  static fromBytes(profile, raw) {
    if (raw.toString('latin1', 0, 4) !== 'DHS0') throw new Error('not a stored secret key');
    const len = raw.readUInt32LE(4);
    if (raw.toString('utf8', 8, 8 + len) !== profile.digest || raw.length !== 8 + len + profile.n) {
      throw new Error('stored secret key is for another profile');
    }
    const c = new Int8Array(profile.n);
    for (let j = 0; j < profile.n; j++) c[j] = raw[8 + len + j] - 1;
    return new SecretKey(profile, c);
  }

  // The secret seen through X -> X^galEl^-1 (NTT-domain permutation), as Lattigo builds Galois keys.
  permuted(galEl) {
    const idx = automorphismIndex(this.p.logN, modInvGaloisElement(galEl, this.p.logN));
    const out = new Map();
    return {
      at: (g) => {
        let v = out.get(g);
        if (!v) {
          const src = this.at(g);
          v = new Float64Array(src.length);
          for (let j = 0; j < src.length; j++) v[j] = src[idx[j]];
          out.set(g, v);
        }
        return v;
      },
    };
  }
}

// --- evaluation keys (raw-u32 wire) ---

// genKey: the gadget encryption of sIn (NTT, Q limbs) under sOut (NTT, Q and P limbs).
function genKey(p, kind, galEl, sIn, sOut, rng) {
  const ring = ringOf(p);
  const n = p.n;
  const mods = [...ring.Q, ...ring.P];
  const QP = mods.length;
  const raw = Buffer.alloc(RAW_KEY_HEADER + 4 * p.digits * 2 * QP * n);
  raw.writeUInt32LE(RAW_KEY_MAGIC, 0);
  raw.writeUInt32LE(kind, 4);
  raw.writeBigUInt64LE(BigInt(galEl), 8);
  raw.writeUInt32LE(p.maxLevel, 16);
  raw.writeUInt32LE(p.pCount - 1, 20);
  raw.writeUInt32LE(p.digits, 24);
  raw.writeUInt32LE(p.logN, 28);
  const body = new Uint32Array(raw.buffer, raw.byteOffset + RAW_KEY_HEADER, p.digits * 2 * QP * n);
  const P = p.p.reduce((x, y) => x * BigInt(y), 1n);
  for (let d = 0; d < p.digits; d++) {
    const e = sampleGaussian(rng, n);
    for (let g = 0; g < QP; g++) {
      const mod = mods[g];
      const q = mod.q;
      const iq = 1 / q;
      const bOff = ((d * 2 + 0) * QP + g) * n;
      const aOff = ((d * 2 + 1) * QP + g) * n;
      const eN = smallNTT(mod, e);
      const so = sOut.at(g);
      const inDigit = g < p.qCount && Math.floor(g / p.pCount) === d;
      const pq = inDigit ? Number(P % BigInt(q)) : 0;
      const si = inDigit ? sIn(g) : null;
      for (let j = 0; j < n; j++) {
        const a = rng.below(q);
        body[aOff + j] = a;
        // b = -a*sOut + e (+ P*sIn on this digit's limbs)
        let b = eN[j] - mulmodInv(a, so[j], q, iq);
        if (b < 0) b += q;
        if (inDigit) {
          b += mulmodInv(pq, si[j], q, iq);
          if (b >= q) b -= q;
        }
        body[bOff + j] = b;
      }
    }
  }
  return raw;
}

export function relinearizationKey(sk, rng = new RandomStream()) {
  const p = sk.p;
  const sq = (g) => {
    const s = sk.at(g);
    const q = p.q[g];
    const out = new Float64Array(s.length);
    for (let j = 0; j < s.length; j++) out[j] = mulmod(s[j], s[j], q);
    return out;
  };
  return genKey(p, 0, 0, sq, sk, rng);
}

export function galoisKey(sk, galEl, rng = new RandomStream()) {
  return genKey(sk.p, 1, galEl, (g) => sk.at(g), sk.permuted(galEl), rng);
}

// Every evaluation key of the profile, one at a time (each is ~79 MB at N = 2^16): relinearization, then one Galois
// key per pinned rotation.
export function* evaluationKeys(sk) {
  const rng = new RandomStream();
  yield relinearizationKey(sk, rng);
  for (const g of sk.p.galois) yield galoisKey(sk, g, rng);
}

// --- encryption and decryption ---

// encryptCompact: symmetric encryption of a compact plaintext (limbs 0..level) at `scale`.
export function encrypt(sk, compact, level, scale, rng = new RandomStream()) {
  const body = new Uint32Array(2 * (level + 1) * sk.p.n);
  encryptLimbs(sk, compact, level, sampleGaussian(rng, sk.p.n), body, 0, level + 1, rng);
  return encodeCt(sk.p, { deg: 1, level, scale, body });
}

// encryptLimbs: limbs [i0, i1) of (c0, c1) = (-a*s + e + pt, a) into body (c0 limbs, then c1 limbs). The error e is
// shared by all limbs (it is one small polynomial); a is fresh per limb.
export function encryptLimbs(sk, compact, level, e, body, i0, i1, rng = new RandomStream()) {
  const p = sk.p;
  const ring = ringOf(p);
  const n = p.n;
  const block = n / COMPACT_WORDS;
  for (let i = i0; i < i1; i++) {
    const mod = ring.Q[i];
    const q = mod.q;
    const iq = 1 / q;
    const eN = smallNTT(mod, e);
    const s = sk.at(i);
    const c0 = i * n;
    const c1 = (level + 1 + i) * n;
    for (let j = 0; j < n; j++) {
      const a = rng.below(q);
      body[c1 + j] = a;
      let b = eN[j] - mulmodInv(a, s[j], q, iq) + compact[i * COMPACT_WORDS + Math.floor(j / block)];
      if (b < 0) b += q;
      else if (b >= q) b -= q;
      body[c0 + j] = b;
    }
  }
}

// sealValues: encode 64 slot values at the default scale and top level, then encrypt.
export function sealValues(sk, values, level = sk.p.maxLevel, rng) {
  const scale = 2 ** sk.p.logScale;
  const compact = encodeCompact(sk.p, values, new BigFloat(128).setFloat64(scale), level);
  return encrypt(sk, compact, level, scale, rng);
}

// decrypt: a raw-u32 ciphertext of degree 1 -> 64 slot values (float64).
export function decrypt(sk, raw) {
  const p = sk.p;
  const ct = decodeCt(p, raw);
  if (ct.deg !== 1) throw new Error('only degree-1 ciphertexts are decrypted');
  const ring = ringOf(p);
  const n = p.n;
  const limbs = [];
  for (let i = 0; i <= ct.level; i++) {
    const mod = ring.Q[i];
    const q = mod.q;
    const s = sk.at(i);
    const row = new Float64Array(n);
    const c0 = ct.body.subarray(i * n, (i + 1) * n);
    const c1 = ct.body.subarray((ct.level + 1 + i) * n, (ct.level + 2 + i) * n);
    for (let j = 0; j < n; j++) {
      const v = c0[j] + mulmod(c1[j], s[j], q);
      row[j] = v >= q ? v - q : v;
    }
    limbs.push(mod.intt(row));
  }
  return decodeCoeffs(p, limbs, ct.level, ct.scale);
}

export { sampleGaussian, RandomStream, COMPACT_WORDS, LOG_SLOTS, RAW_CT_HEADER, RAW_CT_MAGIC, ROTATIONS };
