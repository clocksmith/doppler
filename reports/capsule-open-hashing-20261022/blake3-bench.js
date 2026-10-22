import * as oldB from './old.js';
import * as newB from './new.js';
const hex = async (m, b) => { const h = await m.hash(b); return typeof h === 'string' ? h : Buffer.from(h).toString('hex'); };
let seed = 5; const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) & 255;
for (const n of [0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 3073, 65536, 100003]) {
  const b = Uint8Array.from({ length: n }, rnd);
  if (await hex(oldB, b) !== await hex(newB, b)) throw new Error('mismatch ' + n);
}
const big = Uint8Array.from({ length: 64 << 20 }, rnd);
for (const [name, m] of [['old', oldB], ['new', newB]]) {
  await hex(m, big.subarray(0, 4 << 20)); const t = performance.now(); const d = await hex(m, big); const ms = performance.now() - t;
  console.log(name, (64 / (ms / 1000)).toFixed(0), 'MB/s', d.slice(0, 16));
}
