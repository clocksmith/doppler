// Key generation, encryption and decryption in JavaScript, checked through the homomorphic operations that use the
// keys: a rotation (Galois key) and a multiplication with relinearization (relinearization key) on the CPU executor,
// decrypted and compared with the plaintext result.

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BigFloat } from '../ckks/bigfloat.js';
import { decodeCt, decodeKey, encodeCt } from '../ckks/codec.js';
import { encodeCompact } from '../ckks/encoder.js';
import { SecretKey, decrypt, encrypt, galoisKey, relinearizationKey, sealValues } from '../ckks/keys.js';
import { lookupProfile } from '../ckks/params.js';
import { RnsExec } from '../ckks/rns-exec.js';
import { SLOTS } from '../plan.js';

const p = lookupProfile('dev-insecure-n12-u32');
const sk = SecretKey.generate(p);
const vals = Array.from({ length: SLOTS }, (_, i) => Math.sin(i + 1) * 2);
const close = (a, b, tol) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < tol, `slot ${i}: ${v} vs ${b[i]}`));

test('secret key: ternary, serialized for its profile only', () => {
  const coeffs = [...sk.coeffs];
  assert.ok(coeffs.every((c) => c >= -1 && c <= 1));
  const nonzero = coeffs.filter((c) => c !== 0).length / coeffs.length;
  assert.ok(nonzero > 0.6 && nonzero < 0.73, `density ${nonzero}`);
  const back = SecretKey.fromBytes(p, sk.toBytes());
  assert.deepEqual([...back.coeffs], coeffs);
  assert.throws(() => SecretKey.fromBytes(lookupProfile('research-n16-u32'), sk.toBytes()), /another profile/);
});

test('encryption is randomized and decrypts at the top and bottom levels', () => {
  for (const level of [p.maxLevel, 1]) {
    const a = sealValues(sk, vals, level);
    const b = sealValues(sk, vals, level);
    assert.notDeepEqual(a, b);
    close(decrypt(sk, a), vals, 1e-6);
  }
});

test('rotation with a JavaScript Galois key', () => {
  const x = new RnsExec(p);
  const level = 3;
  const ct = decodeCt(p, sealValues(sk, vals, level));
  for (const [k, gal] of [[1, p.galois[0]], [8, p.galois[3]]]) {
    const raw = galoisKey(sk, gal);
    const key = decodeKey(p, raw, 1);
    x.loadKey(1, key.gal, key.body);
    const out = x.run({ op: 'rotate', level, deg: 1, gal }, ct);
    const got = decrypt(sk, encodeCt(p, { ...out, scale: ct.scale }));
    close(got, vals.map((_, i) => vals[(i + k) % SLOTS]), 1e-5);
  }
});

test('multiplication with a JavaScript relinearization key, then rescale', () => {
  const x = new RnsExec(p);
  const key = decodeKey(p, relinearizationKey(sk), 0);
  x.loadKey(0, 0, key.body);
  const level = 4;
  const scale = 2 ** p.logScale;
  const other = vals.map((v, i) => Math.cos(i) - v / 3);
  const a = decodeCt(p, sealValues(sk, vals, level));
  const b = decodeCt(p, encrypt(sk, encodeCompact(p, other, new BigFloat(128).setFloat64(scale), level), level, scale));
  const t = x.run({ op: 'tensor', level, deg: 2 }, a, b);
  const r = x.run({ op: 'relin', level, deg: 1 }, t);
  const s = x.run({ op: 'rescale', level, deg: 1 }, r);
  const got = decrypt(sk, encodeCt(p, { ...s, scale: (scale * scale) / p.q[level] }));
  close(got, vals.map((v, i) => v * other[i]), 1e-4);
});

test('evaluation keys have the pinned gadget shape and canonical residues', () => {
  const raw = relinearizationKey(sk);
  const key = decodeKey(p, raw, 0);
  assert.equal(key.body.length, p.digits * 2 * (p.qCount + p.pCount) * p.n);
  const bad = Buffer.from(raw);
  bad.writeUInt32LE(p.q[0], 32);
  assert.throws(() => decodeKey(p, bad, 0), /below its modulus/);
});
