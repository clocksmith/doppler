import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runAttention, recordAttention } from '../../src/gpu/kernels/attention.js';

assert.ok((await probeNodeGPU({ installFileFetchShim: true })).ready);
const device = getDevice(), owned = [], errors = [], cases = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const path = { id: 'query_splitk_operator', activationDtype: 'f32', kvDtype: 'f32' };
const step = { op: 'attention', kernel: 'attention_prefill_query_splitk_f32.wgsl', entry: 'main' };
path.prefill = { steps: [step] }; path.decode = { steps: [step] };
const upload = values => {
  const buffer = acquireBuffer(Math.max(16, values.byteLength)); owned.push(buffer); uploadData(buffer, values); return buffer;
};
try {
  for (const [T, S, H, KV] of [[1, 1, 2, 1], [4, 4, 6, 1], [3, 31, 6, 1], [4, 173, 6, 1], [4, 2303, 24, 4], [9, 41, 8, 1], [17, 65, 8, 2]]) {
    const D = 256, start = S - T, amplitude = S === 173 ? 2 : 0.125;
    const q = Float32Array.from({ length: T * H * D }, (_, i) => Math.sin(i * 0.17) * amplitude);
    const k = Float32Array.from({ length: S * KV * D }, (_, i) => Math.cos(i * 0.13) * amplitude);
    const v = Float32Array.from({ length: k.length }, (_, i) => Math.sin(i * 0.11) * 0.25);
    const Q = createTensor(upload(q), 'f32', [T, H, D]);
    const K = createTensor(upload(k), 'f32', [S, KV, D]);
    const V = createTensor(upload(v), 'f32', [S, KV, D]);
    // Independent double-precision masked softmax reference, integral GQA including ratio six.
    const expected = new Float32Array(q.length);
    for (let row = 0; row < T; row++) for (let head = 0; head < H; head++) {
      const keys = start + row + 1, kvHead = Math.floor(head / (H / KV));
      const scores = new Float64Array(keys); let maximum = -Infinity, denominator = 0;
      for (let key = 0; key < keys; key++) {
        let dot = 0;
        for (let d = 0; d < D; d++) dot += q[(row * H + head) * D + d] * k[(key * KV + kvHead) * D + d];
        scores[key] = dot / 16; maximum = Math.max(maximum, scores[key]);
      }
      for (let key = 0; key < keys; key++) denominator += (scores[key] = Math.exp(scores[key] - maximum));
      for (let d = 0; d < D; d++) {
        let value = 0;
        for (let key = 0; key < keys; key++) value += scores[key] * v[(key * KV + kvHead) * D + d];
        expected[(row * H + head) * D + d] = value / denominator;
      }
    }
    for (const recorded of [false, true]) for (const gpuLength of [false, true]) {
      const recorder = recorded ? createCommandRecorder('query_splitk_reference') : null;
      const out = upload(new Float32Array(q.length + 4).fill(-12345));
      const options = { seqLen: T, kvLen: S, numKVHeads: KV, causal: true, startPos: start,
        kernelPath: path, outputBuffer: out, slidingWindow: gpuLength ? null : 0, kvLenBuffer: gpuLength ? upload(new Uint32Array([S])) : null };
      try {
        if (recorder) { await recordAttention(recorder, Q, K, V, null, H, D, options); await recorder.submitAndWait(); }
        else await runAttention(Q, K, V, null, H, D, options);
        const actual = new Float32Array(await readBuffer(out, (q.length + 4) * 4)); let maxAbs = 0;
        for (let i = 0; i < expected.length; i++) {
          const delta = Math.abs(actual[i] - expected[i]); maxAbs = Math.max(maxAbs, delta);
          assert.ok(Number.isFinite(delta) && delta <= 1e-6 + 1e-5 * Math.abs(expected[i]), `${T}/${S}/${H}/${i}: ${delta}`);
        }
        for (let i = q.length; i < actual.length; i++) assert.equal(actual[i], -12345);
        cases.push({ T, S, H, KV, recorded, gpuLength, maxAbs });
      } finally { await recorder?.abort(); }
    }
    if (T === 3) {
      const borrowed = upload(new Float32Array(q.length));
      for (const recorded of [false, true]) for (const failAt of [1, 2]) for (const supplied of [null, borrowed]) {
        const before = getBufferPool().getStats().resources.active.bytes;
        const create = device.createBindGroup; let calls = 0;
        const recorder = recorded ? createCommandRecorder('query_splitk_failure') : null;
        device.createBindGroup = function(descriptor) {
          if (++calls === failAt) throw new Error('injected split-K binding failure');
          return create.call(this, descriptor);
        };
        try {
          const options = { seqLen: T, kvLen: S, numKVHeads: KV, causal: true, startPos: start,
            kernelPath: path, outputBuffer: supplied };
          await assert.rejects(recorded ? recordAttention(recorder, Q, K, V, null, H, D, options)
            : runAttention(Q, K, V, null, H, D, options), /injected split-K binding failure/);
        } finally { device.createBindGroup = create; await recorder?.abort(); }
        assert.equal(getBufferPool().getStats().resources.active.bytes, before, 'failure releases owned buffers and retains borrowed output');
      }
    }
    for (const patch of [{ slidingWindow: 8 }, { attnSoftcap: 1 }, { causal: false }, { bidirectionalSpanLength: 1 }, { outputBuffer: Q.buffer }]) {
      await assert.rejects(runAttention(Q, K, V, null, H, D, { seqLen: T, kvLen: S, numKVHeads: KV,
        causal: true, startPos: start, kernelPath: path, ...patch }));
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: true, cases, errors, adapter: getKernelCapabilities().adapterInfo,
    scope: 'F32 split-K attention operator reference, modes, GQA, tails and bounds; not full-model performance' }));
} finally {
  await device.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
