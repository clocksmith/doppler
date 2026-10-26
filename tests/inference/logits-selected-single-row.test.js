import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, getBufferPool, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { computeLogits } from '../../src/inference/pipelines/text/logits/index.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const device = getDevice();
  const hidden = acquireBuffer(32, undefined, 'selected_single_hidden');
  const norm = acquireBuffer(16, undefined, 'selected_single_norm');
  const head = acquireBuffer(32, undefined, 'selected_single_head');
  device.queue.writeBuffer(hidden, 0, Float32Array.from([1, 2, 3, 4, 4, 3, 2, 1]));
  device.queue.writeBuffer(norm, 0, new Float32Array(4).fill(1));
  // Exact binary16 identity matrix: the selected logits equal normalized inputs.
  device.queue.writeBuffer(head, 0, Uint16Array.from([
    0x3c00, 0, 0, 0, 0, 0x3c00, 0, 0, 0, 0, 0x3c00, 0, 0, 0, 0, 0x3c00,
  ]));
  const weights = { finalNorm: createWeightBuffer(norm, 'f32', 'row', [4]),
    lmHead: createWeightBuffer(head, 'f16', 'row', [4, 4]) };
  const config = { hiddenSize: 4, vocabSize: 4, rmsNormEps: 1e-6, rmsNormWeightOffset: false,
    normalizationType: 'rmsnorm', activationDtype: 'f32', useTiedEmbeddings: false,
    finalLogitSoftcapping: null, logitInputScale: 1, logitOutputScale: 1 };
  const execute = (count, lastPositionOnly) => computeLogits(hidden, count, weights, config,
    true, {}, undefined, undefined, undefined, { lastPositionOnly, selectedTokenIds: [1, 3] });
  try {
    const before = getBufferPool().getStats().activeBuffers;
    for (const [count, expected] of [[1, [2, 4]], [2, [3, 1]]]) {
      const actual = await execute(count, true);
      assert.equal(actual.length, 2);
      for (let i = 0; i < 2; i++) {
        assert.ok(Math.abs(actual[i] - expected[i] / Math.sqrt(7.5 + 1e-6)) < 1e-5);
      }
      assert.equal(getBufferPool().getStats().activeBuffers, before);
    }
    await assert.rejects(execute(2, false), /requires exactly one hidden row/);
    assert.equal(getBufferPool().getStats().activeBuffers, before);
  } finally {
    releaseBuffer(hidden); releaseBuffer(norm); releaseBuffer(head);
  }
}
destroyDevice();
console.log(`logits-selected-single-row: ${probe.ready ? 'GPU checks passed' : `skipped: ${probe.reason}`}`);
