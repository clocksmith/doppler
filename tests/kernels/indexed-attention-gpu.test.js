import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { recordIndexedAttention, runIndexedAttention } from '../../src/gpu/kernels/qsa-indexer.js';

// Contract: attention over QSA-selected cache rows equals float64 softmax
// attention restricted to those rows within a derived F32 bound, for the dense
// prefix mode and the listed mode, at the qwen4_exp geometry (24 heads, 2 KV
// heads, head dim 256, f16 KV), immediate and recorded. Excluded rows have no
// influence. Synthetic exact inputs; not learned-weight evidence.
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical indexed attention check unavailable: ${probe.reason}`);
const capabilities = getKernelCapabilities();
assert.ok(!/swiftshader|llvmpipe|lavapipe/.test(JSON.stringify(capabilities?.adapterInfo ?? {}).toLowerCase()));

const u = 2 ** -24;
const gamma = (n) => (n * u) / (1 - n * u);
const numHeads = 24;
const numKVHeads = 2;
const headDim = 256;
const kvRows = 2600;
const maxSelected = 2051;
const scale = 1 / 16;
// Deterministic exact values: f16-representable keys/values, f32 queries.
const hash = (i, salt) => {
  let x = (Math.imul(i + 1, 0x9E3779B1) ^ Math.imul(salt, 0x85EBCA77)) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x2C1B3C6D) >>> 0;
  return ((x ^ (x >>> 12)) >>> 0) / 2 ** 32;
};
const f16Exact = (i, salt) => (Math.round(hash(i, salt) * 2048) - 1024) / 512;   // k / 512 in [-2, 2)
const toF16 = (value) => {
  // Exact conversion for values k / 512 with |k| <= 1024 (normal f16 range).
  if (value === 0) return 0;
  const sign = value < 0 ? 0x8000 : 0;
  const magnitude = Math.abs(value);
  const exponent = Math.floor(Math.log2(magnitude));
  const mantissa = Math.round((magnitude / 2 ** exponent - 1) * 1024);
  return sign | ((exponent + 15) << 10) | mantissa;
};
const kv = (salt) => {
  const values = Float64Array.from({ length: kvRows * numKVHeads * headDim }, (_, i) => f16Exact(i, salt));
  return { values, bits: Uint16Array.from(values, toF16) };
};
const keys = kv(1);
const vals = kv(2);
const cases = [
  { name: 'dense prefix', count: 37, mode: 0, rows: null },
  { name: 'dense at limit', count: 2051, mode: 0, rows: null },
  { name: 'listed sparse', count: 2049, mode: 1,
    rows: Array.from({ length: 2600 }, (_, i) => i).filter((i) => hash(i, 9) < 0.79 || i >= 2596).slice(-2049) },
];
for (const entry of cases) if (entry.rows) assert.equal(entry.rows.length, entry.count);
const numTokens = cases.length;
const query = Float32Array.from({ length: numTokens * numHeads * headDim }, (_, i) => Math.fround((hash(i, 3) - 0.5) * 4));

function reference(token) {
  const { count, rows } = cases[token];
  const selected = rows ?? Array.from({ length: count }, (_, i) => i);
  const value = new Float64Array(numHeads * headDim);
  const bound = new Float64Array(numHeads * headDim);
  for (let h = 0; h < numHeads; h++) {
    const kvh = Math.floor(h / (numHeads / numKVHeads));
    const q = query.subarray((token * numHeads + h) * headDim, (token * numHeads + h + 1) * headDim);
    const scores = selected.map((row) => {
      let dot = 0;
      let magnitude = 0;
      for (let d = 0; d < headDim; d++) {
        const k = keys.values[(row * numKVHeads + kvh) * headDim + d];
        dot += q[d] * k;
        magnitude += Math.abs(q[d] * k);
      }
      return { s: dot * scale, beta: (gamma(headDim) * magnitude + u * Math.abs(dot)) * scale };
    });
    const m = Math.max(...scores.map((x) => x.s));
    const range = m - Math.min(...scores.map((x) => x.s));
    const betaMax = Math.max(...scores.map((x) => x.beta));
    const chunks = Math.ceil(selected.length / 256);
    const weights = scores.map((x) => Math.exp(x.s - m));
    const z = weights.reduce((a, b) => a + b, 0);
    // Log-relative error of each unnormalized weight: score error, max error, subtraction and exp
    // rounding, plus one rescaling factor per chunk (difference rounding, exp, fma).
    const eta = 2 * betaMax + u * range + 8 * u + chunks * (u * range + 10 * u);
    const normalized = 2 * eta + gamma(selected.length) + 2 * u;
    for (let d = 0; d < headDim; d++) {
      let out = 0;
      let absolute = 0;
      selected.forEach((row, j) => {
        const v = vals.values[(row * numKVHeads + kvh) * headDim + d];
        out += (weights[j] / z) * v;
        absolute += (weights[j] / z) * Math.abs(v);
      });
      value[h * headDim + d] = out;
      bound[h * headDim + d] = (gamma(selected.length + chunks) + Math.expm1(normalized)) * absolute + 2 * u * Math.abs(out);
    }
  }
  return { value, bound };
}

const owned = new Set();
const upload = (data) => {
  const buffer = acquireBuffer(data.byteLength);
  owned.add(buffer);
  uploadData(buffer, data);
  return buffer;
};
const results = [];
try {
  const keyBuffer = upload(keys.bits);
  const valueBuffer = upload(vals.bits);
  const selection = new Uint32Array(numTokens * maxSelected);
  const meta = new Uint32Array(numTokens * 2);
  cases.forEach((entry, t) => {
    meta[t * 2] = entry.count;
    meta[t * 2 + 1] = entry.mode;
    if (entry.rows) selection.set(entry.rows, t * maxSelected);
    else selection.fill(0xffffffff, t * maxSelected, (t + 1) * maxSelected); // dense mode never reads the list
  });
  const selectionBuffers = { selection: upload(selection), meta: upload(meta), maxSelected };
  const q = createTensor(upload(query), 'f32', [numTokens, numHeads * headDim]);
  const expected = cases.map((_, t) => reference(t));
  for (const mode of ['immediate', 'recorded']) {
    const recorder = mode === 'recorded' ? new CommandRecorder(getDevice(), 'indexed_attention_check') : null;
    const options = { numTokens, numHeads, numKVHeads, headDim, scale, kvRows };
    const output = recorder
      ? await recordIndexedAttention(recorder, q, keyBuffer, valueBuffer, selectionBuffers, options)
      : await runIndexedAttention(q, keyBuffer, valueBuffer, selectionBuffers, options);
    owned.add(output.buffer);
    if (recorder) await recorder.submitAndWait();
    const actual = new Float32Array(await readBuffer(output.buffer, numTokens * numHeads * headDim * 4));
    cases.forEach((entry, t) => {
      let worst = 0;
      for (let i = 0; i < numHeads * headDim; i++) {
        const error = Math.abs(actual[t * numHeads * headDim + i] - expected[t].value[i]);
        worst = Math.max(worst, error / expected[t].bound[i]);
      }
      assert.ok(Number.isFinite(worst) && worst <= 1, `${mode}/${entry.name}: error/bound=${worst}`);
      results.push({ mode, case: entry.name, ratio: worst });
    });
  }
  // Discrimination: attending to all 2600 rows instead of the listed 2049 exceeds the bound.
  const listed = 2;
  const full = { ...cases[listed] };
  cases[listed] = { name: 'all rows', count: kvRows, mode: 0, rows: null };
  const unselected = reference(listed);
  cases[listed] = full;
  let separation = Infinity;
  for (let i = 0; i < numHeads * headDim; i++) {
    separation = Math.min(separation, expected[listed].bound[i] / Math.max(Math.abs(unselected.value[i] - expected[listed].value[i]), 1e-300));
  }
  assert.ok(separation < 0.5, 'the listed-mode case distinguishes selected from all rows');
  await assert.rejects(runIndexedAttention(q, keyBuffer, valueBuffer, selectionBuffers,
    { numTokens, numHeads, numKVHeads: 5, headDim, scale, kvRows }), /multiple of numKVHeads/);
  await assert.rejects(runIndexedAttention(q, keyBuffer, valueBuffer, selectionBuffers,
    { numTokens, numHeads, numKVHeads, headDim, scale, kvRows: kvRows * 2 }), /f16 key cache/);
} finally {
  await getDevice()?.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
console.log(JSON.stringify({ test: 'indexed-attention-gpu', adapter: capabilities?.adapterInfo ?? null, results,
  scope: 'Physical indexed attention arithmetic on synthetic exact inputs; no learned weights or timing' }));
