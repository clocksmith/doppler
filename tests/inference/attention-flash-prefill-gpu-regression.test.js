import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runAttention, recordAttention } from '../../src/gpu/kernels/attention.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { f32ToF16Array } from '../../src/inference/kv-cache/types.js';

const probe = await probeNodeGPU();
if (!probe.ready) {
  console.log(`attention-flash-prefill: skipped (${probe.reason})`);
  process.exit(0);
}
const kernelPath = {
  id: 'flash-prefill-regression',
  activationDtype: 'f32',
  prefill: { steps: [{ op: 'attention', kernel: 'attention_prefill_flash_head256_f16kv.wgsl', entry: 'main' }] },
};
const headDim = 256;
const numHeads = 2;
const numKVHeads = 1;

async function runCase(seqLen, kvLen, recorded) {
  const q = Float32Array.from({ length: seqLen * numHeads * headDim }, (_, i) => Math.sin(i * 0.017) * 0.25);
  const k = Float32Array.from({ length: kvLen * headDim }, (_, i) => ((i * 17 % 31) - 15) / 32);
  const v = Float32Array.from({ length: kvLen * headDim }, (_, i) => ((i * 7 % 23) - 11) / 16);
  const buffers = [];
  function tensor(values, dtype, shape) {
    const buffer = acquireBuffer(values.byteLength);
    buffers.push(buffer);
    uploadData(buffer, values);
    return createTensor(buffer, dtype, shape);
  }
  const Q = tensor(q, 'f32', [seqLen, numHeads, headDim]);
  const K = tensor(f32ToF16Array(k), 'f16', [kvLen, numKVHeads, headDim]);
  const V = tensor(f32ToF16Array(v), 'f16', [kvLen, numKVHeads, headDim]);
  const options = { seqLen, kvLen, numKVHeads, causal: true, startPos: kvLen - seqLen, kernelPath };
  const device = getDevice();
  device.pushErrorScope('validation');
  let output;
  let scopeOpen = true;
  try {
    if (recorded) {
      const recorder = new CommandRecorder(device, 'flash-prefill-regression');
      output = await recordAttention(recorder, Q, K, V, null, numHeads, headDim, options);
      await recorder.submitAndWait();
    } else {
      output = await runAttention(Q, K, V, null, numHeads, headDim, options);
      await device.queue.onSubmittedWorkDone();
    }
    const error = await device.popErrorScope();
    scopeOpen = false;
    assert.equal(error, null, error?.message);
    const actual = new Float32Array(await readBuffer(output.buffer));
    let maxAbs = 0;
    for (let row = 0; row < seqLen; row++) {
      const keys = options.startPos + row + 1;
      for (let head = 0; head < numHeads; head++) {
        const offset = (row * numHeads + head) * headDim;
        const scores = new Float64Array(keys);
        for (let key = 0; key < keys; key++) {
          let score = 0;
          for (let dim = 0; dim < headDim; dim++) score += q[offset + dim] * k[key * headDim + dim];
          scores[key] = score / Math.sqrt(headDim);
        }
        const max = Math.max(...scores);
        let denominator = 0;
        for (let key = 0; key < keys; key++) denominator += (scores[key] = Math.exp(scores[key] - max));
        for (let dim = 0; dim < headDim; dim++) {
          let expected = 0;
          for (let key = 0; key < keys; key++) expected += scores[key] * v[key * headDim + dim];
          expected /= denominator;
          const delta = Math.abs(actual[offset + dim] - expected);
          assert(Number.isFinite(delta) && delta < 1e-4, `row=${row} head=${head} dim=${dim}: ${delta}`);
          maxAbs = Math.max(maxAbs, delta);
        }
      }
    }
    await assert.rejects(() => runAttention(Q, K, V, null, numHeads, headDim, {
      ...options, bidirectionalSpanStart: 0, bidirectionalSpanLength: 1,
    }), /[Ff]lash.*bidirectional/);
    if (kvLen > 64) {
      for (const failAt of [1, 2]) {
        for (const supplied of [null, output.buffer]) {
          const activeBefore = getBufferPool().getStats().resources.active.bytes;
          const createBindGroup = device.createBindGroup;
          const recorder = recorded ? new CommandRecorder(device, 'flash-failure') : null;
          let calls = 0;
          device.createBindGroup = function(descriptor) {
            if (++calls === failAt) throw new Error('injected flash preparation failure');
            return createBindGroup.call(this, descriptor);
          };
          try {
            const failOptions = { ...options, outputBuffer: supplied };
            await assert.rejects(() => recorded
              ? recordAttention(recorder, Q, K, V, null, numHeads, headDim, failOptions)
              : runAttention(Q, K, V, null, numHeads, headDim, failOptions), /injected flash preparation failure/);
          } finally {
            device.createBindGroup = createBindGroup;
            await recorder?.abort();
          }
          assert.equal(getBufferPool().getStats().resources.active.bytes, activeBefore,
            `Leaked or released borrowed buffers: recorded=${recorded}, failAt=${failAt}, supplied=${!!supplied}`);
        }
      }
    }
    console.log(JSON.stringify({ seqLen, kvLen, recorded, maxAbs }));
  } finally {
    if (scopeOpen) await device.popErrorScope();
    if (output) releaseBuffer(output.buffer);
    buffers.forEach(releaseBuffer);
  }
}

try {
  // Single split, multiple splits, query tails, and an existing KV prefix.
  for (const recorded of [false, true]) {
    await runCase(31, 31, recorded);
    await runCase(65, 129, recorded);
  }
} finally {
  await destroyDevice();
}
