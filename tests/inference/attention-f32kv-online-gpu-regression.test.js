import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runAttention, recordAttention } from '../../src/gpu/kernels/attention.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  console.log(`attention-f32kv-online: skipped (${probe.reason})`);
  process.exit(0);
}
const device = getDevice();
const receipt = { timeUTC: new Date().toISOString(), probe, capabilities: getKernelCapabilities(),
  scope: 'Component numerical and instrumented GPU timing only; no full-model or cross-hardware performance claim.', cases: [], timings: [] };
const save = () => {
  if (process.env.DOPPLER_ATTENTION_TEST_RECEIPT) {
    writeFileSync(process.env.DOPPLER_ATTENTION_TEST_RECEIPT, JSON.stringify(receipt, null, 2));
  }
};
function kernelPath(kernel) {
  return { id: 'f32kv-online-component', activationDtype: 'f32',
    decode: { steps: [{ op: 'attention', kernel, entry: 'main' }] } };
}
function reference(q, k, v, spec) {
  const { kvLen, headDim, numHeads, numKVHeads, startPos, kvStart, slidingWindow, attnSoftcap } = spec;
  const result = new Float64Array(numHeads * headDim);
  for (let head = 0; head < numHeads; head++) {
    const kvHead = Math.floor(head / (numHeads / numKVHeads));
    const scores = new Float64Array(kvLen);
    let maximum = -Infinity;
    for (let key = 0; key < kvLen; key++) {
      const absolute = kvStart + key;
      if (absolute > startPos || (slidingWindow && absolute < startPos - slidingWindow + 1)) {
        scores[key] = -Infinity;
        continue;
      }
      let dot = 0;
      for (let dim = 0; dim < headDim; dim++) dot += q[head * headDim + dim] * k[(key * numKVHeads + kvHead) * headDim + dim];
      let score = dot / Math.sqrt(headDim);
      if (attnSoftcap) score = Math.tanh(score / attnSoftcap) * attnSoftcap;
      scores[key] = score;
      maximum = Math.max(maximum, score);
    }
    let sum = 0;
    for (let key = 0; key < kvLen; key++) sum += (scores[key] = Math.exp(scores[key] - maximum));
    for (let dim = 0; dim < headDim; dim++) {
      let value = 0;
      for (let key = 0; key < kvLen; key++) value += scores[key] * v[(key * numKVHeads + kvHead) * headDim + dim];
      result[head * headDim + dim] = sum > 0 ? value / sum : 0;
    }
  }
  return result;
}
async function runCase(input, recorded) {
  const spec = { numHeads: 16, numKVHeads: 2, headDim: 256, kvStart: 0, slidingWindow: 0, attnSoftcap: 0, ...input };
  spec.startPos = spec.kvStart + Math.max(0, spec.kvLen - 1);
  const { kvLen, numHeads, numKVHeads, headDim } = spec;
  const q = Float32Array.from({ length: numHeads * headDim }, (_, i) => Math.sin(i * 0.017) * 0.25);
  const k = Float32Array.from({ length: kvLen * numKVHeads * headDim }, (_, i) => ((i * 17 % 31) - 15) / 32);
  const v = Float32Array.from({ length: k.length }, (_, i) => ((i * 7 % 23) - 11) / 16);
  const owned = [];
  function tensor(values, shape) {
    const buffer = acquireBuffer(Math.max(4, values.byteLength));
    owned.push(buffer);
    if (values.byteLength) uploadData(buffer, values);
    return createTensor(buffer, 'f32', shape);
  }
  const Q = tensor(q, [1, numHeads, headDim]);
  const K = tensor(k, [kvLen, numKVHeads, headDim]);
  const V = tensor(v, [kvLen, numKVHeads, headDim]);
  let output;
  let recorder;
  device.pushErrorScope('validation');
  let scopeOpen = true;
  try {
    const options = { ...spec, seqLen: 1, causal: true, kernelPath: kernelPath('attention_decode_online.wgsl') };
    if (recorded) {
      recorder = new CommandRecorder(device, 'f32kv-online-reference');
      output = await recordAttention(recorder, Q, K, V, null, numHeads, headDim, options);
      await recorder.submitAndWait();
    } else {
      output = await runAttention(Q, K, V, null, numHeads, headDim, options);
      await device.queue.onSubmittedWorkDone();
    }
    const error = await device.popErrorScope();
    scopeOpen = false;
    assert.equal(error, null, error?.message);
    const actual = new Float32Array(await readBuffer(output.buffer, numHeads * headDim * 4));
    const expected = reference(q, k, v, spec);
    let maxAbs = 0;
    for (let i = 0; i < expected.length; i++) {
      const delta = Math.abs(actual[i] - expected[i]);
      assert(Number.isFinite(delta) && delta < 1e-4, `kvLen=${kvLen} headDim=${headDim} recorded=${recorded} index=${i}: ${delta}`);
      maxAbs = Math.max(maxAbs, delta);
    }
    receipt.cases.push({ ...spec, recorded, maxAbs });
    save();
    console.log(JSON.stringify(receipt.cases.at(-1)));
    if (recorded && !spec.slidingWindow && kvLen > 0 && headDim === 256) {
      for (let repeat = 0; repeat < 6; repeat++) {
        const order = repeat % 2 ? ['attention_decode_online.wgsl', 'attention_streaming.wgsl'] : ['attention_streaming.wgsl', 'attention_decode_online.wgsl'];
        for (const kernel of order) {
          const timedRecorder = new CommandRecorder(device, 'f32kv-component-timing', { profile: true, profilePhase: 'decode' });
          if (!timedRecorder.isProfilingEnabled()) { await timedRecorder.abort(); continue; }
          const timed = await recordAttention(timedRecorder, Q, K, V, null, numHeads, headDim, { ...options, kernelPath: kernelPath(kernel) });
          try {
            await timedRecorder.submitAndWait();
            const timings = await timedRecorder.resolveProfileTimings({ requireComplete: true });
            receipt.timings.push({ kvLen, kernel, repeat, warmup: repeat === 0,
              gpuMs: Object.values(timings).reduce((sum, value) => sum + value, 0), recorder: timedRecorder.getStats() });
          } finally { await timedRecorder.abort(); releaseBuffer(timed.buffer); }
        }
      }
      save();
    }
  } finally {
    await recorder?.abort();
    if (scopeOpen) await device.popErrorScope();
    if (output) releaseBuffer(output.buffer);
    owned.forEach(releaseBuffer);
  }
}
try {
  for (const recorded of [false, true]) {
    for (const spec of [
      { kvLen: 0 }, { kvLen: 1 }, { kvLen: 43 }, { kvLen: 257 }, { kvLen: 2287 },
      { kvLen: 73, slidingWindow: 17, attnSoftcap: 2 },
      { kvLen: 129, numHeads: 4, numKVHeads: 2, headDim: 128 },
      { kvLen: 129, numHeads: 4, numKVHeads: 2, headDim: 512 },
    ]) await runCase(spec, recorded);
  }
} finally { save(); await destroyDevice(); }
