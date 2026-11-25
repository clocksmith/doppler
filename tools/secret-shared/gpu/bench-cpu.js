#!/usr/bin/env node
/**
 * Ring-matvec throughput of the plain-JS CPU operator path (ring-cpu.js) at the 27B MLP shape (gate/up: 17408 x 5120)
 * on random Q4_K blocks and uniform 64-bit shares, the same case as the Go operator's BenchmarkQ4KMatVec27B.
 *
 *   node bench-cpu.js [--threads N] [--rows 17408] [--cols 5120] [--reps 5]
 */
import { parseArgs } from 'node:util';
import { createCpuRingMatVec } from './ring-cpu.js';

const { values: a } = parseArgs({ options: { threads: { type: 'string', default: '0' }, rows: { type: 'string', default: '17408' },
  cols: { type: 'string', default: '5120' }, reps: { type: 'string', default: '5' } } });
const rows = Number(a.rows), cols = Number(a.cols);
const q4k = new Uint8Array(rows * cols / 256 * 144);
let s = 1;
for (let i = 0; i < q4k.length; i++) q4k[i] = (s = (s * 1103515245 + 12345) >>> 0) >>> 24;
const view = new DataView(q4k.buffer);
for (let o = 0; o < q4k.length; o += 144) { view.setUint16(o, 0x1400 | (o % 1023), true); view.setUint16(o + 2, 0x1000 | (o % 1021), true); }
const x = new Uint32Array(2 * cols).map(() => (s = (s * 1103515245 + 12345) >>> 0));
const ring = createCpuRingMatVec({ fw: 24, threads: Number(a.threads) });
const t = ring.tensor(rows, cols, q4k);
await ring.run([t], x, 1, rows);
const times = [];
for (let r = 0; r < Number(a.reps); r++) { const t0 = performance.now(); await ring.run([t], x, 1, rows); times.push(performance.now() - t0); }
times.sort((p, q) => p - q);
const ms = times[times.length >> 1];
console.log(JSON.stringify({ shape: [rows, cols], threads: ring.threads, medianMs: +ms.toFixed(2), gmacPerS: +(rows * cols / ms / 1e6).toFixed(2), node: process.version }));
ring.close();
