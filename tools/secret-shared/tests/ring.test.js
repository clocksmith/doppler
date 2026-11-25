// node --test tools/secret-shared/tests/   (plain CPU; opens no GPU device)
import assert from 'node:assert/strict';
import test from 'node:test';
import { q4kRows, denseRows } from '../gpu/ring-cpu.js';
import { encode, encodeAndShare, recombine, rint } from '../requester/ring.js';
import { shake256 } from '../requester/node.js';

const M64 = (1n << 64n) - 1n;
const words = v => [Number(v & 0xffffffffn), Number(v >> 32n)];
const fromWords = (w, i) => (BigInt(w[2 * i + 1]) << 32n) | BigInt(w[2 * i]);
function lcg(seed) { let s = BigInt(seed); return () => (s = (s * 6364136223846793005n + 1442695040888963407n) & M64); }
function f16(bits) { const s = bits & 0x8000 ? -1 : 1, e = (bits >> 10) & 31, f = bits & 1023; return e === 0 ? s * f * 2 ** -24 : s * (1 + f / 1024) * 2 ** (e - 15); }
const roundAway = v => (v < 0 ? -Math.round(-v) : Math.round(v));

test('CPU Q4_K ring matvec equals a BigInt reference, mod 2^64, on uniform shares', () => {
  const next = lcg(7), rows = 9, cols = 512, n = 3, fw = 24;
  const q4k = new Uint8Array(rows * (cols / 256) * 144);
  for (let i = 0; i < q4k.length; i++) q4k[i] = Number(next() >> 56n);
  const view = new DataView(q4k.buffer);
  for (let o = 0; o < q4k.length; o += 144) { view.setUint16(o, 0x1400 | Number(next() >> 54n), true); view.setUint16(o + 2, 0x1000 | Number(next() >> 54n), true); }
  const xs = Array.from({ length: n * cols }, () => next());
  const x = new Uint32Array(xs.flatMap(words));
  const y = new Uint32Array(n * rows * 2);
  q4kRows(q4k, cols, x, n, 0, rows, fw, y, rows, 0);
  for (let c = 0; c < n; c++) for (let r = 0; r < rows; r++) {
    let acc = 0n;
    for (let b = 0; b < cols / 256; b++) {
      const o = (r * cols / 256 + b) * 144, d = f16(view.getUint16(o, true)), dmin = f16(view.getUint16(o + 2, true));
      for (let sb = 0; sb < 8; sb++) {
        const s = q4k.subarray(o + 4, o + 16);
        const sc = sb < 4 ? s[sb] & 63 : (s[sb + 4] & 0xf) | ((s[sb - 4] >> 6) << 4);
        const m = sb < 4 ? s[sb + 4] & 63 : (s[sb + 4] >> 4) | ((s[sb] >> 6) << 4);
        const A = BigInt(roundAway(d * sc * 2 ** fw)), B = BigInt(roundAway(dmin * m * 2 ** fw));
        for (let l = 0; l < 32; l++) {
          const byte = q4k[o + 16 + 32 * (sb >> 1) + l], q = BigInt(sb % 2 ? byte >> 4 : byte & 15);
          acc += (A * q - B) * xs[c * cols + b * 256 + 32 * sb + l];
        }
      }
    }
    assert.equal(fromWords(y, c * rows + r), acc & M64, `row ${r} column ${c}`);
  }
});

test('CPU dense ring rows equal a BigInt reference with signed fixed-point weights', () => {
  const next = lcg(9), rows = 4, cols = 300, n = 2;
  const w = Float64Array.from({ length: rows * cols }, (_, i) => (i % 7 === 0 ? 0 : Number(BigInt.asIntN(32, next() >> 32n))));
  const xs = Array.from({ length: n * cols }, () => next());
  const y = new Uint32Array(n * rows * 2);
  denseRows(w, cols, new Uint32Array(xs.flatMap(words)), n, 0, rows, y, rows, 0);
  for (let c = 0; c < n; c++) for (let r = 0; r < rows; r++) {
    let acc = 0n;
    for (let k = 0; k < cols; k++) acc += BigInt(w[r * cols + k]) * xs[c * cols + k];
    assert.equal(fromWords(y, c * rows + r), acc & M64);
  }
});

test('encoding is numpy rint(ldexp(x, fx)) in two\'s complement; sharing and 64-bit recombination are exact', () => {
  assert.deepEqual([rint(2.5), rint(-2.5), rint(3.5), rint(-1.5), rint(0.49)], [2, -2, 4, -2, 0]);
  const x = Float32Array.from([0, 1.5, -1.5, 3.14159, -1e-7, 1234.5678, -65504]);
  const e = encode(x, 20);
  x.forEach((v, i) => assert.equal(BigInt.asIntN(64, fromWords(e, i)), BigInt(rint(v * 2 ** 20))));
  const next = lcg(3), x0 = new Uint32Array(Array.from(x).flatMap(() => words(next()))), x1 = new Uint32Array(2 * x.length);
  encodeAndShare(x, 20, x0, x1);
  x.forEach((_, i) => assert.equal((fromWords(x0, i) + fromWords(x1, i)) & M64, fromWords(e, i)));
  // Output shares y0 + y1 = y at scale 2^-(fx+fw): 64-bit recombination returns float32(float64(y) * 2^-44) exactly.
  const ys = [0n, 1n << 44n, -(3n << 43n), 123456789012345n, -(1n << 62n) + 12345n];
  const y0 = ys.map(() => next()), y1 = ys.map((v, i) => (v - y0[i]) & M64);
  const out = new Float32Array(ys.length);
  recombine(new Uint8Array(new Uint32Array(y0.flatMap(words)).buffer), new Uint8Array(new Uint32Array(y1.flatMap(words)).buffer), 8, 0, 20, 24, out);
  ys.forEach((v, i) => assert.equal(out[i], Math.fround(Number(v) * 2 ** -44)));
});

test('packed returns (bytes shift/8..7 of each share) recombine to within 2^shift of the true value', () => {
  const next = lcg(5);
  for (const shift of [8, 16, 24, 32]) {
    const width = 8 - shift / 8, count = 2000, a = new Uint8Array(width * count), b = new Uint8Array(width * count), truth = [];
    for (let i = 0; i < count; i++) {
      const y = BigInt.asIntN(64, next()) >> BigInt(next() % 40n), y0 = next(), y1 = (y - y0) & M64;
      truth.push(y);
      for (let k = 0; k < width; k++) { a[width * i + k] = Number((y0 >> BigInt(8 * (k + shift / 8))) & 0xffn); b[width * i + k] = Number((y1 >> BigInt(8 * (k + shift / 8))) & 0xffn); }
    }
    const out = new Float64Array(count);
    recombine(a, b, width, shift, 0, 0, out);
    truth.forEach((y, i) => assert.ok(Math.abs(out[i] - Number(y)) <= 2 ** shift * (1 + 1e-12) + Math.abs(Number(y)) * 2 ** -52, `shift ${shift}: ${out[i]} vs ${y}`));
  }
});

test('PRG vectors: SHAKE-256 matches Python hashlib; AES-256-CTR (WebCrypto) matches openssl enc -aes-256-ctr', async () => {
  const seed = Uint8Array.from({ length: 32 }, (_, i) => i);
  assert.equal(Buffer.from(shake256(seed, 7, 48)).toString('hex'),
    'dd0566df0a3f090d2fa8e2cc92164d9b96d3fff9904c3a6202b1fee26302e39b2dfe220c451965f9a75151caa7a9ecf7');
  const key = await crypto.subtle.importKey('raw', seed, 'AES-CTR', false, ['encrypt']);
  const counter = new Uint8Array(16); new DataView(counter.buffer).setBigUint64(0, 7n, false);
  const out = Buffer.from(await crypto.subtle.encrypt({ name: 'AES-CTR', counter, length: 64 }, key, new Uint8Array(48)));
  assert.equal(out.toString('hex'), '3754fb01bdf925d28f8d964fce69304d0486426f0df0dfa8e09791abbb162ea0e275cd65f84287563f0eeba3038d83c5');
});
