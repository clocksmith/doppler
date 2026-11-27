// Pinned CKKS profiles of the protected route. A profile fixes the ring, the RNS moduli (Lattigo's NTT-friendly
// primes and primitive roots, pinned in profiles.json), the slot layout and the protocol identity. The requester and
// the evaluator each load the same table; neither accepts a profile from the other side.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { LOG_SLOTS } from '../plan.js';
import { Modulus, isPrime } from './ring.js';
import { BigFloat } from './bigfloat.js';

export const ROTATIONS = [1, 2, 4, 8, 16, 32];
export const WIRE_RAW_U32 = 'raw-u32/v0';

const TABLE = JSON.parse(readFileSync(new URL('./profiles.json', import.meta.url), 'utf8'));

export function profileIds() {
  return Object.keys(TABLE.profiles);
}

function goList(v) {
  return `[${v.join(' ')}]`;
}

// The profile digest exactly as the Go tools computed it (the evaluators and receipts carry it).
export function profileDigest(p) {
  if (p.digestVersion === 2) {
    const parameters = [p.id, p.logN, p.q, p.p, p.psi, p.galois, p.logScale, LOG_SLOTS, p.rotations, p.wire, p.insecure];
    return `sha256:${createHash('sha256').update(JSON.stringify(parameters)).digest('hex')}`;
  }
  let s = `${p.id}|${p.logN}|${goList(p.logQ)}|${goList(p.logP)}|${p.logScale}|${LOG_SLOTS}|${goList(ROTATIONS)}`;
  if (p.wire) s += `|${p.wire}`;
  return `sha256:${createHash('sha256').update(s).digest('hex')}`;
}

const cache = new Map();

// lookupProfile: the profile with its moduli checked (prime, NTT-friendly for 2N, below 2^32, of the pinned size).
export function lookupProfile(id) {
  if (cache.has(id)) return cache.get(id);
  const t = TABLE.profiles[id];
  if (!t) throw new Error(`unknown profile ${JSON.stringify(id)}`);
  const p = { id, rotations: ROTATIONS, ...t };
  const n = 1 << p.logN;
  const check = (list, bits, what) => {
    if (list.length !== bits.length) throw new Error(`${id}: ${what} count does not match the profile`);
    list.forEach((m, i) => {
      if (!(m < 2 ** 32) || m % (2 * n) !== 1 || !isPrime(m) || Math.abs(Math.log2(m) - bits[i]) > 1) {
        throw new Error(`${id}: ${what}[${i}] = ${m} is not a ${bits[i]}-bit NTT prime for N = ${n}`);
      }
    });
  };
  check(p.q, p.logQ, 'Q');
  check(p.p, p.logP, 'P');
  if (new Set([...p.q, ...p.p]).size !== p.q.length + p.p.length) throw new Error(`${id}: moduli repeat`);
  if (!p.rotations.length || new Set(p.rotations).size !== p.rotations.length || p.rotations.length !== p.galois.length
    || p.rotations.some(rotation => !ROTATIONS.includes(rotation))) throw new Error(`${id}: invalid rotation key schedule`);
  if (profileDigest(p) !== p.digest) throw new Error(`${id}: profile digest does not match its pinned value`);
  p.n = n;
  p.maxLevel = p.q.length - 1;
  p.qCount = p.q.length;
  p.pCount = p.p.length;
  p.digits = Math.ceil(p.qCount / p.pCount);
  p.defaultScale = 2 ** p.logScale;
  cache.set(id, p);
  return p;
}

// Ring context: one Modulus (with NTT tables) per Q and P prime. Built lazily: tables at N = 2^16 take a moment.
export function ringOf(p) {
  if (!p.ring) {
    p.ring = {
      Q: p.q.map((q, i) => new Modulus(q, p.psi[i], p.logN)),
      P: p.p.map((q, i) => new Modulus(q, p.psi[p.qCount + i], p.logN)),
    };
  }
  return p.ring;
}

// Lattigo's NewScale: a 128-bit big float.
export function scaleOf(x) {
  const z = new BigFloat(128);
  if (x instanceof BigFloat) return z.set(x);
  if (typeof x === 'bigint') return z.setInt(x);
  return z.setFloat64(x);
}

export function defaultScale(p) {
  return scaleOf(2 ** p.logScale);
}
