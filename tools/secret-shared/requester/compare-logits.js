#!/usr/bin/env node
/**
 * Compares per-step logits (.npy, float32 [steps, vocab], as written by requester.js --logits-out) against references:
 * the worst absolute difference per prompt and overall, and top-1 agreement.
 *
 *   node compare-logits.js PREFIX REF_PREFIX [REF_PREFIX ...] [--prompts 5]
 *   e.g. node compare-logits.js js/T3-forced logits/control logits/ss-forced
 */
import fs from 'node:fs';
import { parseArgs } from 'node:util';

const { positionals, values } = parseArgs({ allowPositionals: true, options: { prompts: { type: 'string', default: '5' } } });
const [prefix, ...refs] = positionals;

function readNpy(file) {
  const b = fs.readFileSync(file);
  const len = b.readUInt16LE(8), header = b.toString('latin1', 10, 10 + len);
  if (!header.includes("'<f4'")) throw Error(`${file}: not float32`);
  const shape = header.match(/\(([^)]*)\)/)[1].split(',').filter(Boolean).map(Number);
  const data = new Float32Array(b.buffer.slice(b.byteOffset + 10 + len, b.byteOffset + b.length));
  return { rows: shape[0], width: shape[1], data };
}

const argmax = (d, o, w) => { let best = 0; for (let i = 1; i < w; i++) if (d[o + i] > d[o + best]) best = i; return best; };
const report = {};
for (const ref of refs) {
  let worst = 0, agree = 0, total = 0;
  const perPrompt = [];
  for (let p = 0; p < Number(values.prompts); p++) {
    const a = readNpy(`${prefix}-${p}.npy`), b = readNpy(`${ref}-${p}.npy`);
    const rows = Math.min(a.rows, b.rows);
    let w = 0;
    for (let r = 0; r < rows; r++) {
      const o = r * a.width;
      for (let i = 0; i < a.width; i++) { const d = Math.abs(a.data[o + i] - b.data[o + i]); if (d > w) w = d; }
      if (argmax(a.data, o, a.width) === argmax(b.data, o, b.width)) agree++;
      total++;
    }
    perPrompt.push({ prompt: p, positions: rows, maxAbs: w });
    if (w > worst) worst = w;
  }
  report[ref] = { worstMaxAbs: worst, top1: `${agree}/${total}`, perPrompt };
}
console.log(JSON.stringify(report, null, 1));
