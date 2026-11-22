import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, getBufferPool, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { computeLogits } from '../../src/inference/pipelines/text/logits/index.js';

// trailingRows projects only the final N hidden rows; each must equal the matching row
// of the complete projection (same kernels, same weights).
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const device = getDevice();
  const hidden = acquireBuffer(48, undefined, 'test_hidden');
  const norm = acquireBuffer(16, undefined, 'test_norm');
  const head = acquireBuffer(80, undefined, 'test_head');
  device.queue.writeBuffer(hidden, 0, Float32Array.from([1, 2, 3, 4, 4, 3, 2, 1, -1, 0.5, 2, -3]));
  device.queue.writeBuffer(norm, 0, new Float32Array(4).fill(1));
  device.queue.writeBuffer(head, 0, Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, -1, 1, -1]));
  const weights = { finalNorm: createWeightBuffer(norm, 'f32', 'row', [4]), lmHead: createWeightBuffer(head, 'f32', 'row', [5, 4]) };
  const config = { hiddenSize: 4, vocabSize: 5, rmsNormEps: 1e-6, rmsNormWeightOffset: false,
    normalizationType: 'rmsnorm', activationDtype: 'f32', useTiedEmbeddings: false,
    finalLogitSoftcapping: null, logitInputScale: 1, logitOutputScale: 1 };
  const execute = options => computeLogits(hidden, 3, weights, config, true, {}, undefined, undefined, undefined, options);
  try {
    const before = getBufferPool().getStats().activeBuffers;
    const full = await execute({ lastPositionOnly: false });
    assert.equal(full.length, 15);
    const trailing = await execute({ trailingRows: 2 });
    assert.equal(trailing.length, 10);
    assert.deepEqual(Array.from(trailing), Array.from(full.subarray(5)), 'trailing rows match the complete projection');
    const all = await execute({ trailingRows: 3 });
    assert.deepEqual(Array.from(all), Array.from(full));
    const last = await execute({ trailingRows: 1 });
    assert.deepEqual(Array.from(last), Array.from(full.subarray(10)));
    await assert.rejects(execute({ trailingRows: 4 }), /trailingRows must be/);
    await assert.rejects(execute({ trailingRows: 2, selectedTokenIds: [1] }), /excludes/);
    assert.equal(getBufferPool().getStats().activeBuffers, before, 'every temporary is released');
  } finally { releaseBuffer(hidden); releaseBuffer(norm); releaseBuffer(head); }
}
destroyDevice();
console.log(`logits-trailing-rows: ${probe.ready ? 'GPU checks passed' : `skipped: ${probe.reason}`}`);
