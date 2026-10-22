import { createHash } from 'node:crypto';
import { createSha256Hasher } from './old.js';
import { compressBlock } from './new-compress.js';
const IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
function newHash(bytes) { // full-block path + same padding as old.js
  const st = new Int32Array(IV), w = new Int32Array(64);
  let o = 0; for (; o + 64 <= bytes.length; o += 64) compressBlock(st, w, bytes, o);
  const tail = bytes.length - o, pad = new Uint8Array(tail < 56 ? 64 : 128); pad.set(bytes.subarray(o)); pad[tail] = 0x80;
  const bits = BigInt(bytes.length) * 8n; for (let i = 0; i < 8; i++) pad[pad.length - 8 + i] = Number((bits >> BigInt(56 - i * 8)) & 0xffn);
  for (let p = 0; p < pad.length; p += 64) compressBlock(st, w, pad, p);
  return Array.from(st, (v) => (v >>> 0).toString(16).padStart(8, '0')).join('');
}
let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) & 255;
for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 65536, 65539]) {
  const b = Uint8Array.from({ length: n }, rnd); const ref = createHash('sha256').update(b).digest('hex');
  const o = createSha256Hasher(); o.update(b);
  if (o.digestHex() !== ref || newHash(b) !== ref) throw new Error('mismatch at ' + n);
}
const big = Uint8Array.from({ length: 64 * 1024 * 1024 }, rnd);
for (const [name, fn] of [['old', (x) => { const h = createSha256Hasher(); for (let o = 0; o < x.length; o += 65536) h.update(x.subarray(o, o + 65536)); return h.digestHex(); }], ['new', newHash]]) {
  fn(big.subarray(0, 4 << 20)); const t = performance.now(); const d = fn(big); const ms = performance.now() - t;
  console.log(name, (64 / (ms / 1000)).toFixed(0), 'MB/s', d === createHash('sha256').update(big).digest('hex'));
}
