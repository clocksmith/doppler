import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runLinearAttentionCoreGPU } from '../../src/gpu/kernels/linear-attention-core.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
const probe = await probeNodeGPU();
if (!probe.ready) { console.log(`register-decode-gpu: skipped (${probe.reason})`); process.exit(0); }
const silu = x => x / (1 + Math.exp(-x));
const close = (a, b, tolerance, label) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Number.isFinite(a[i]) && Math.abs(a[i] - b[i]) <= tolerance,
    `${label}[${i}]: ${a[i]} != ${b[i]}`);
};
async function check(T, V, window = 1, recorded = false) {
  const K = 128, C = K * 2 + V;
  const qkv = Float32Array.from({ length: T * C }, (_, i) => Math.sin(i * 0.037) * 0.2);
  const z = Float32Array.from({ length: T * V }, (_, i) => Math.cos(i * 0.013) * 0.25);
  const a = Float32Array.from({ length: T }, (_, i) => Math.sin(i) * 0.3);
  const b = Float32Array.from({ length: T }, (_, i) => Math.cos(i) * 0.2);
  const state = Float32Array.from({ length: K * V }, (_, i) => Math.sin(i * 0.007) * 0.005);
  const initialConv = Float32Array.from({ length: C * window }, (_, i) => Math.sin(i * 0.021) * 0.05);
  const convWeights = Float32Array.from({ length: C * window }, (_, i) => Math.cos(i * 0.013) * 0.5);
  const expectedConv = initialConv.slice();
  const expectedState = state.slice(), expected = new Float32Array(T * V);
  for (let t = 0; t < T; t++) {
    const conv = new Float32Array(C);
    for (let channel = 0; channel < C; channel++) {
      const base = channel * window;
      for (let k = 0; k + 1 < window; k++) expectedConv[base + k] = expectedConv[base + k + 1];
      expectedConv[base + window - 1] = qkv[t * C + channel];
      let mixed = 0;
      for (let k = 0; k < window; k++) mixed += expectedConv[base + k] * convWeights[base + k];
      conv[channel] = silu(mixed);
    }
    const q = conv.subarray(0, K), k = conv.subarray(K, 2 * K);
    const qscale = 1 / Math.sqrt(K) / Math.sqrt(q.reduce((s, x) => s + x * x, 0) + 1e-6);
    const kscale = 1 / Math.sqrt(k.reduce((s, x) => s + x * x, 0) + 1e-6);
    const beta = 1 / (1 + Math.exp(-b[t]));
    const decay = Math.exp(-Math.exp(-0.5) * Math.log1p(Math.exp(a[t] + 0.1)));
    const row = new Float32Array(V);
    for (let v = 0; v < V; v++) {
      let kv = 0;
      for (let d = 0; d < K; d++) {
        const i = d * V + v;
        expectedState[i] = expectedState[i] * decay;
        kv = Math.fround(expectedState[i] * Math.fround(k[d] * kscale) + kv);
      }
      const delta = Math.fround((conv[2 * K + v] - kv) * beta);
      let out = 0;
      for (let d = 0; d < K; d++) {
        const i = d * V + v;
        expectedState[i] = Math.fround(Math.fround(k[d] * kscale) * delta + expectedState[i]);
        out = Math.fround(expectedState[i] * Math.fround(q[d] * qscale) + out);
      }
      row[v] = out;
    }
    const inv = 1 / Math.sqrt(row.reduce((s, x) => s + x * x, 0) / V + 1e-6);
    for (let v = 0; v < V; v++) expected[t * V + v] = row[v] * inv * silu(z[t * V + v]);
  }
  async function run(enabled) {
    setRuntimeConfig({ inference: { session: { useLinearAttentionFusedDecodeCore: true, useLinearAttentionRegisterDecode: enabled } } });
    const buffers = [];
    const upload = values => { const buffer = acquireBuffer(values.byteLength); uploadData(buffer, values); buffers.push(buffer); return buffer; };
    const layer = { convDim: C, vSize: V, convKernelSize: window, numVHeads: 1, numKHeads: 1,
      headKDim: K, headVDim: V, valueDim: V, qSize: K, kSize: K, qRep: 1,
      normMode: 'shared', rmsNormEps: 1e-6, recurrentStatePrecision: 'f32', recurrentStateComponents: 1,
      convWeightGPU: upload(convWeights), convStateGPU: upload(initialConv),
      dtBiasGPU: upload(new Float32Array([0.1])), aLogGPU: upload(new Float32Array([-0.5])),
      normWeightGPU: upload(new Float32Array(V).fill(1)), recurrentStateGPU: upload(state) };
    let result;
    try {
      const actual = new Float32Array(T * V);
      for (let t = 0; t < T; t++) {
        const q = createTensor(upload(qkv.slice(t * C, (t + 1) * C)), 'f32', [1, C]);
        const gate = createTensor(upload(z.slice(t * V, (t + 1) * V)), 'f32', [1, V]);
        const ab = createTensor(upload(new Float32Array([a[t], b[t]])), 'f32', [1, 2]);
        const recorder = recorded ? new CommandRecorder(null, 'register_decode_reference') : null;
        result = await runLinearAttentionCoreGPU(q, gate, ab, ab, layer, {
          numTokens: 1, normGateActivation: 'silu', outputDtype: 'f32', qkL2NormEps: 1e-6,
          abPacked: true, bProjOffsetElements: 1, recorder,
        });
        if (recorder) await recorder.submitAndWait();
        actual.set(new Float32Array(await readBuffer(result.buffer, V * 4)), t * V);
        releaseBuffer(result.buffer); result = null;
      }
      return { output: actual,
        state: new Float32Array(await readBuffer(layer.recurrentStateGPU, state.byteLength)),
        convState: new Float32Array(await readBuffer(layer.convStateGPU, initialConv.byteLength)) };

    } finally { if (result) releaseBuffer(result.buffer); for (const buffer of buffers) releaseBuffer(buffer); }
  }
  const baseline = await run(false), candidate = await run(true);
  close(candidate.output, baseline.output, 1e-5, 'baseline output');
  close(candidate.state, baseline.state, 1e-6, 'baseline state');
  close(candidate.convState, baseline.convState, 0, 'baseline convolution state');
  close(candidate.convState, expectedConv, 0, 'CPU convolution state');
  close(candidate.output, expected, 1e-4, 'CPU output');
  close(candidate.state, expectedState, 1e-5, 'CPU state');
}
try {
  for (const recorded of [false, true])
  for (const [T, V, window] of [[1, 16, 1], [7, 128, 1], [31, 128, 1], [1, 128, 4], [7, 128, 4], [31, 128, 4]]) await check(T, V, window, recorded);
  console.log('register-decode-gpu: passed 12 immediate/recorded sequential baseline and CPU-reference cases, including nonzero convolution history');
} finally { resetRuntimeConfig(); await destroyDevice(); }
