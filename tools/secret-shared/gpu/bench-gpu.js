#!/usr/bin/env node
/** Ring matvec throughput at a 27B MLP shape (17408 x 5120 Q4_K, one decode column), GPU kernel plus readback. */
import { randomFillSync } from 'node:crypto';
import { openDevice, createRingMatVec } from './ring-matvec.js';

const rows = 17408, cols = 5120, iterations = Number(process.argv[2] ?? 20);
const { device, adapterInfo } = await openDevice();
const ring = createRingMatVec(device, { fw: 24 });
const q4k = randomFillSync(new Uint8Array(rows * cols / 256 * 144));
for (let i = 0; i < q4k.length; i += 144) { q4k[i + 1] = 0x14; q4k[i + 3] = 0x10; }  // modest f16 scales
const tensor = ring.tensor(rows, cols, q4k);
const x = randomFillSync(new Uint32Array(cols * 2));
await ring.run([tensor], x, 1, rows);
const t0 = performance.now();
for (let i = 0; i < iterations; i++) await ring.run([tensor], x, 1, rows);
const ms = (performance.now() - t0) / iterations;
console.log(JSON.stringify({ adapter: adapterInfo, shape: [rows, cols], msPerMatvec: +ms.toFixed(2), gmacPerS: +((rows * cols) / ms / 1e6).toFixed(2) }));
process.exit(0);
