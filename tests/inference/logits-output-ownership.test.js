import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, getBufferPool, releaseBuffer, readBuffer } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { computeLogits } from '../../src/inference/pipelines/text/logits/index.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const device = getDevice();
  const hidden = acquireBuffer(32, undefined, 'test_hidden');
  const norm = acquireBuffer(16, undefined, 'test_norm');
  const head = acquireBuffer(64, undefined, 'test_head');
  device.queue.writeBuffer(hidden, 0, Float32Array.from([1, 2, 3, 4, 4, 3, 2, 1]));
  device.queue.writeBuffer(norm, 0, new Float32Array(4).fill(1));
  device.queue.writeBuffer(head, 0, Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
  const weights = { finalNorm: createWeightBuffer(norm, 'f32', 'row', [4]), lmHead: createWeightBuffer(head, 'f32', 'row', [4, 4]) };
  const config = { hiddenSize: 4, vocabSize: 4, rmsNormEps: 1e-6, rmsNormWeightOffset: false,
    normalizationType: 'rmsnorm', activationDtype: 'f32', useTiedEmbeddings: false,
    finalLogitSoftcapping: null, logitInputScale: 1, logitOutputScale: 1 };
  const execute = (overrides, options) => computeLogits(hidden, 2, { ...weights, ...overrides }, config,
    true, {}, undefined, undefined, undefined, { lastPositionOnly: true, ...options });
  try {
    const before = getBufferPool().getStats().activeBuffers;
    await assert.rejects(execute({ lmHeadBias: new Float32Array([1]) }, { returnGpuBuffer: true }), /bias requires/);
    assert.equal(getBufferPool().getStats().activeBuffers, before, 'failed finalization releases every temporary; input and weights stay borrowed');
    const canonical = await execute({}, {});
    assert.equal(getBufferPool().getStats().activeBuffers, before);
    const result = await execute({}, { returnGpuBuffer: true });
    assert.equal(getBufferPool().getStats().activeBuffers, before + 1, 'only returned GPU logits transfer to caller');
    try {
      assert.deepEqual(new Float32Array(await readBuffer(result.logitsBuffer, 16)), canonical);
    } finally { releaseBuffer(result.logitsBuffer); }
    assert.equal(getBufferPool().getStats().activeBuffers, before);
  } finally { releaseBuffer(hidden); releaseBuffer(norm); releaseBuffer(head); }
}
destroyDevice();
console.log(`logits-output-ownership: ${probe.ready ? 'GPU checks passed' : `skipped: ${probe.reason}`}`);
