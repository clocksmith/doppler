import assert from 'node:assert/strict';
import { assertSubgroupPartition } from '../../src/gpu/kernels/subgroup-partition.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runAttention, recordAttention } from '../../src/gpu/kernels/attention.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { f32ToF16Array } from '../../src/inference/kv-cache/types.js';

const probe = await probeNodeGPU();
if (!probe.ready) {
  console.log(`attention-partitioned-prefill: skipped (${probe.reason})`);
  process.exit(0);
}
const capabilities = getKernelCapabilities();
try {
  assert.equal(capabilities.hasF16, true, 'requires shader-f16');
  assert.equal(capabilities.hasSubgroups, true, 'requires subgroups');
  assertSubgroupPartition(16, capabilities.adapterInfo);
} catch (error) {
  console.log('attention-partitioned-prefill: skipped (' + error.message + ')');
  await destroyDevice();
  process.exit(0);
}
const kernelPath = {
  id: 'partitioned-prefill-regression',
  activationDtype: 'f32',
  prefill: { steps: [{ op: 'attention', kernel: 'attention_prefill_partitioned_head256_f16kv.wgsl', entry: 'main' }] },
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
      const recorder = new CommandRecorder(device, 'partitioned-prefill-regression');
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
    await runCase(17, 655, recorded);
  }
} finally {
  await destroyDevice();
}
