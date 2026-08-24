import assert from 'node:assert/strict';

import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import {
  acquireBuffer,
  destroyBufferPool,
  getBufferPool,
  readBuffer,
  releaseBuffer,
} from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runFinalizeLogitsTensor } from '../../src/gpu/kernels/logit-finalize.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const gpuProbe = await probeNodeGPU({ installFileFetchShim: true });

if (gpuProbe.ready) {
  const device = getDevice();
  const inputValues = new Float32Array(2 * 3);
  const inputBuffer = acquireBuffer(inputValues.byteLength, undefined, 'logits_bias_input');
  device.queue.writeBuffer(inputBuffer, 0, inputValues);
  const input = createTensor(inputBuffer, 'f32', [2, 3], 'logits_bias_input');

  const biased = await runFinalizeLogitsTensor(input, {
    rowCount: 2,
    sourceColumns: 3,
    targetColumns: 3,
    bias: new Float32Array([0.25, -0.5, 1.75]),
    outputScale: 1,
    softcap: 0,
  });
  const values = new Float32Array(await readBuffer(biased.buffer, 2 * 3 * 4));
  assert.deepEqual(Array.from(values), [
    0.25, -0.5, 1.75,
    0.25, -0.5, 1.75,
  ]);
  releaseBuffer(biased.buffer);

  const scaled = await runFinalizeLogitsTensor(input, {
    rowCount: 2,
    sourceColumns: 3,
    targetColumns: 3,
    bias: new Float32Array([0.25, -0.5, 1.75]),
    outputScale: 0.5,
    softcap: 0,
  });
  const scaledValues = new Float32Array(await readBuffer(scaled.buffer, 2 * 3 * 4));
  assert.deepEqual(Array.from(scaledValues), [
    0.125, -0.25, 0.875,
    0.125, -0.25, 0.875,
  ]);
  releaseBuffer(scaled.buffer);

  const activeBeforeRejection = getBufferPool().getStats().activeBuffers;
  await assert.rejects(
    () => runFinalizeLogitsTensor(input, {
      rowCount: 2,
      sourceColumns: 3,
      targetColumns: 3,
      bias: new Float32Array([1, 2]),
      outputScale: 1,
      softcap: 0,
    }),
    /bias requires at least 3 values/
  );
  assert.equal(getBufferPool().getStats().activeBuffers, activeBeforeRejection);
  releaseBuffer(inputBuffer);
}

destroyBufferPool();
destroyDevice();

console.log(
  gpuProbe.ready
    ? 'logits-bias.test: ok'
    : `logits-bias.test: ok (GPU assertions skipped: ${gpuProbe.reason})`
);
