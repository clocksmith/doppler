// Kernel tests on a real GPU: every RNS operation of the compiled programs, at several levels, must come out of the
// WebGPU kernels word for word as the JavaScript reference executor (ckks/rns-exec.js) computes it. The reference is
// itself pinned to Lattigo by test/golden.test.js; with the golden profile and seed the inputs are those of the
// one-time Go export, so the GPU results are also checked against Lattigo's digests directly.
//
//   node tools/privacy-protected/webgpu/test-vectors.js [--profile dev-insecure-n12-u32] [--seed 0x5eed] [--only rotate]

import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { encodeCt } from '../ckks/codec.js';
import { lookupProfile } from '../ckks/params.js';
import { RnsExec } from '../ckks/rns-exec.js';
import { SplitMix64 } from '../test/splitmix.js';
import { HeDevice, RAW_CT_HEADER, outLevel } from './he-device.js';
import { openDevice } from './gpu.js';
import { prepareProgramBlob } from './rns-tables.js';
import { digest } from './wire.js';

const { values } = parseArgs({ options: {
  profile: { type: 'string', default: 'dev-insecure-n12-u32' }, seed: { type: 'string', default: '0x5eed' }, only: { type: 'string' },
} });
const p = lookupProfile(values.profile);
const golden = JSON.parse(readFileSync(new URL('../fixtures/go-golden.json', import.meta.url), 'utf8')).rns;
const useGolden = golden.profile === p.id && BigInt(golden.seed) === BigInt(values.seed);
const n = p.n;
const mods = [...p.q, ...p.p];
const rng = new SplitMix64(BigInt(values.seed));
const cpu = new RnsExec(p);
const gpu = await openDevice();
console.log(`adapter ${gpu.provider} ${JSON.stringify(gpu.adapter)} profile ${p.id} N=2^${p.logN}${useGolden ? ' (golden inputs)' : ''}`);
const he = new HeDevice(gpu.device, { logN: p.logN, logSlots: 6, q: p.q, p: p.p, psi: p.psi });
let failed = 0;
let passed = 0;

function firstDiff(x, y) {
  const m = Math.min(x.length, y.length);
  for (let i = 0; i < m; i++) if (x[i] !== y[i]) return i;
  return x.length === y.length ? -1 : m;
}

// NTT and inverse NTT under every modulus against the CPU transform (own input stream, so the case inputs below stay
// those of the golden export).
{
  const nttRng = new SplitMix64(BigInt(values.seed) + 1n);
  const input = new Uint32Array(mods.length * n);
  for (let i = 0; i < input.length; i++) input[i] = nttRng.below(mods[Math.floor(i / n)]);
  const ring = cpu.mods;
  const want = new Uint32Array(input.length);
  ring.forEach((m, g) => want.set(m.ntt(Float64Array.from(input.subarray(g * n, (g + 1) * n))), g * n));
  const buf = he.upload(input.slice(), 'ntt-test');
  const ct = { buf, deg: 0, level: mods.length - 1, cs: mods.length * n };
  he.ntt(buf, 0, mods.length, 0);
  const fwd = new Uint32Array((await he.readCt(ct, 1)).buffer.slice(RAW_CT_HEADER));
  he.intt(buf, 0, mods.length, 0);
  const back = new Uint32Array((await he.readCt(ct, 1)).buffer.slice(RAW_CT_HEADER));
  for (const [name, got, w] of [['ntt', fwd, want], ['intt(ntt)', back, input]]) {
    const d = firstDiff(got, w);
    if (d < 0) passed++;
    else { failed++; console.log(`FAIL ${name}: first difference at word ${d} (limb ${Math.floor(d / n)})`); }
  }
}

// Keys: random gadget residues (key switching is linear algebra on them; real keys are not needed to test it).
const keyBody = () => {
  const b = new Uint32Array(p.digits * 2 * mods.length * n);
  let o = 0;
  for (let d = 0; d < p.digits; d++) for (let c = 0; c < 2; c++) for (const q of mods) for (let j = 0; j < n; j++) b[o++] = rng.below(q);
  return b;
};
const rawKey = (kind, gal, body) => {
  const raw = Buffer.alloc(32 + 4 * body.length);
  raw.writeUInt32LE(0x304b4844, 0);
  raw.writeUInt32LE(kind, 4);
  raw.writeBigUInt64LE(BigInt(gal), 8);
  [p.maxLevel, p.pCount - 1, p.digits, p.logN].forEach((w, i) => raw.writeUInt32LE(w, 16 + 4 * i));
  Buffer.from(body.buffer).copy(raw, 32);
  return raw;
};
for (const [kind, gal] of [[0, 0], ...p.galois.map((g) => [1, g])]) {
  const body = keyBody();
  cpu.loadKey(kind, gal, body);
  he.loadKey(rawKey(kind, gal, body));
}

const words = (level, per) => Uint32Array.from({ length: (level + 1) * per }, (_, i) => rng.below(p.q[Math.floor(i / per)]));
const randCt = (deg, level) => {
  const ct = cpu.newCt(deg, level);
  for (let i = 0; i < ct.body.length; i++) ct.body[i] = rng.below(mods[Math.floor(i / n) % (level + 1)]);
  return ct;
};
let ci = 0;
for (const l of [36, 25, 26, 13, 12, 3, 1]) {
  const blobs = { smul: words(l, 1), sadd: words(l, 1), smac: words(l, 1), pmul: words(l, 128), padd: words(l, 128) };
  const cases = [
    ['add', { op: 'add', deg: 2 }, [[1, Math.min(p.maxLevel, l + 2)], [2, l]]], ['sub', { op: 'sub', deg: 2 }, [[1, l], [2, l]]],
    ['smul', { op: 'smul', deg: 2, k: 0 }, [[2, l]]], ['sadd', { op: 'sadd', deg: 1, k: 0 }, [[1, l]]],
    ['smac', { op: 'smac', deg: 2, k: 0 }, [[1, l], [2, l]]], ['pmul', { op: 'pmul', deg: 1, pt: 0 }, [[1, l]]],
    ['padd', { op: 'padd', deg: 1, pt: 0 }, [[1, l]]], ['tensor', { op: 'tensor', deg: 2 }, [[1, l], [1, l]]],
    ['relin', { op: 'relin', deg: 1 }, [[2, l]]], ['rescale', { op: 'rescale', deg: 2 }, [[2, l]]], ['rescale', { op: 'rescale', deg: 1 }, [[1, l]]],
    ...p.galois.map((g, i) => [`rotate-${2 ** i}`, { op: 'rotate', deg: 1, gal: g }, [[1, l]]]),
  ];
  for (const [name, base, shapes] of cases) {
    const g = useGolden ? golden.cases[ci] : null;
    ci++;
    const ins = shapes.map(([d, lv]) => randCt(d, lv));
    if (values.only && !name.startsWith(values.only)) continue;
    const op = { k: -1, pt: -1, ...base, level: l, a: 0, b: ins.length > 1 ? 1 : -1 };
    const blob = blobs[base.op] ?? new Uint32Array(0);
    const want = cpu.run(op, ins[0], ins[1], blob);
    const wantRaw = encodeCt(p, { ...want, scale: 0 });
    if (g && digest(Buffer.from(want.body.buffer)) !== g.digest) {
      failed++;
      console.log(`FAIL ${name}@${l}: the CPU reference differs from Lattigo's digest`);
      continue;
    }
    const gpuBlob = prepareProgramBlob({ q: p.q, ops: [op] }, blob.slice());
    const blobBuf = he.upload(gpuBlob.length ? gpuBlob : new Uint32Array(4), 'case-blob');
    const gins = ins.map((x) => he.uploadCt(encodeCt(p, { ...x, scale: 1 })));
    const out = he.allocCt(op.deg, outLevel(op));
    he.execOp(op, gins[0] ?? null, gins[1] ?? null, out, blobBuf);
    const got = await he.readCt(out, 0);
    const shapeOk = out.deg === want.deg && out.level === want.level;
    const d = firstDiff(got.subarray(RAW_CT_HEADER), wantRaw.subarray(RAW_CT_HEADER));
    if (shapeOk && d < 0) passed++;
    else {
      failed++;
      const w = Math.floor(d / 4);
      console.log(`FAIL ${name}@${l}: shape ${shapeOk ? 'ok' : 'differs'}, first difference at word ${w}`);
    }
    for (const x of gins) he.freeCt(x);
    he.freeCt(out);
    he.destroy(blobBuf);
  }
}
console.log(`${passed} passed, ${failed} failed; dispatches ${he.stats.dispatches}, peak GPU buffers ${(he.stats.peakBytes / 2 ** 20).toFixed(1)} MiB`);
await gpu.close();
process.exit(failed ? 1 : 0);
