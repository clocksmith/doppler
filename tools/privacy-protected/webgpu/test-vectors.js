// Kernel tests: every case from `pp gpu-vectors` (computed by Lattigo) must come out of the GPU byte for byte.
//
//   node tools/privacy-protected/webgpu/test-vectors.js --vectors <dir>

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { HeDevice, RAW_CT_HEADER, outLevel } from './he-device.js';
import { openDevice } from './gpu.js';
import { prepareProgramBlob } from './rns-tables.js';

const { values } = parseArgs({ options: { vectors: { type: 'string' }, only: { type: 'string' } } });
if (!values.vectors) throw new Error('--vectors <dir> is required');
const dir = values.vectors;
const params = JSON.parse(readFileSync(join(dir, 'params.json'), 'utf8'));

function frames(buf, at) {
  const out = [];
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  while (at < buf.byteLength) {
    const len = Number(v.getBigUint64(at, true));
    out.push(buf.subarray(at + 8, at + 8 + len));
    at += 8 + len;
  }
  return out;
}

function firstDiff(x, y) {
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) if (x[i] !== y[i]) return i;
  return x.length === y.length ? -1 : n;
}

const gpu = await openDevice();
console.log(`adapter ${gpu.provider} ${JSON.stringify(gpu.adapter)} profile ${params.profile} N=2^${params.logN}`);
const he = new HeDevice(gpu.device, { logN: params.logN, logSlots: 6, q: params.q, p: params.p, psi: params.psi });
let failed = 0;
let passed = 0;

// NTT and inverse NTT under every modulus.
{
  const raw = readFileSync(join(dir, 'ntt.bin'));
  const words = new Uint32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const n = 1 << params.logN;
  const m = params.q.length + params.p.length;
  const input = new Uint32Array(m * n);
  const expect = new Uint32Array(m * n);
  for (let g = 0; g < m; g++) {
    input.set(words.subarray(2 * g * n, 2 * g * n + n), g * n);
    expect.set(words.subarray(2 * g * n + n, 2 * g * n + 2 * n), g * n);
  }
  const buf = he.upload(input, 'ntt-test');
  const ct = { buf, deg: 0, level: m - 1, cs: m * n };
  he.ntt(buf, 0, m, 0);
  const fwd = new Uint32Array((await he.readCt(ct, 1)).buffer.slice(RAW_CT_HEADER));
  he.intt(buf, 0, m, 0);
  const back = new Uint32Array((await he.readCt(ct, 1)).buffer.slice(RAW_CT_HEADER));
  for (const [name, got, want] of [['ntt', fwd, expect], ['intt(ntt)', back, input]]) {
    const d = firstDiff(got, want);
    if (d < 0) passed++;
    else { failed++; console.log(`FAIL ${name}: first difference at word ${d} (limb ${Math.floor(d / n)}): ${got[d]} vs ${want[d]}`); }
  }
  console.log(`ntt: ${m} moduli checked`);
}

for (const key of frames(readFileSync(join(dir, 'keys.bin')), 0)) he.loadKey(key);

for (const c of params.cases) {
  if (values.only && !c.name.startsWith(values.only)) continue;
  const raw = readFileSync(join(dir, c.file));
  const v = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const words = v.getUint32(0, true);
  const blob = new Uint32Array(raw.buffer.slice(raw.byteOffset + 4, raw.byteOffset + 4 + 4 * words));
  const cts = frames(raw, 4 + 4 * words);
  const op = c.op;
  prepareProgramBlob({ q: params.q, ops: [op] }, blob);
  const blobBuf = he.upload(blob.length ? blob : new Uint32Array(4), 'case-blob');
  const ins = cts.slice(0, c.inputs).map((x) => he.uploadCt(x));
  const want = cts[c.inputs];
  const out = he.allocCt(op.deg, outLevel(op));
  he.execOp(op, ins[0] ?? null, ins[1] ?? null, out, blobBuf);
  const got = await he.readCt(out, 0);
  const wv = new DataView(want.buffer, want.byteOffset, want.byteLength);
  const shapeOk = wv.getUint32(4, true) === out.deg && wv.getUint32(8, true) === out.level;
  const d = firstDiff(got.subarray(RAW_CT_HEADER), want.subarray(RAW_CT_HEADER));
  if (shapeOk && d < 0) {
    passed++;
  } else {
    failed++;
    const n = 1 << params.logN;
    const w = Math.floor(d / 4);
    console.log(`FAIL ${c.name}: shape ${shapeOk ? 'ok' : 'differs'}, first difference at word ${w} (component ${Math.floor(w / out.cs)}, limb ${Math.floor((w % out.cs) / n)}, coefficient ${w % n})`);
  }
  for (const x of ins) he.freeCt(x);
  he.freeCt(out);
  he.destroy(blobBuf);
}
console.log(`${passed} passed, ${failed} failed; dispatches ${he.stats.dispatches}, peak GPU buffers ${(he.stats.peakBytes / 2 ** 20).toFixed(1)} MiB`);
await gpu.close();
process.exit(failed ? 1 : 0);
