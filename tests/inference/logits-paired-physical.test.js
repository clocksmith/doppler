import assert from 'node:assert/strict';
import { snapshotTensor } from '../../src/debug/tensor.js';
import { setGPUDevice } from '../../src/debug/config.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, getBufferPool, readBuffer } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { computeLogits, recordLogitsGPU } from '../../src/inference/pipelines/text/logits/index.js';
import { computeLogitsGPU } from '../../src/inference/pipelines/text/logits/gpu.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const device = getDevice(), hiddenSize = 128, vocabSize = 4;
  setGPUDevice(device);
  const input = new Float32Array(2 * hiddenSize * 2);
  for (let token = 0; token < 2; token++) for (let k = 0; k < hiddenSize; k++) {
    input[(token * hiddenSize + k) * 2] = Math.fround((token ? 1 : -1) * (1 + k / 256));
    input[(token * hiddenSize + k) * 2 + 1] = Math.fround((k % 2 ? -1 : 1) * 2e-8);
  }
  const normValues = Float32Array.from({ length: hiddenSize }, (_, k) => 0.75 + k / 256);
  const headValues = Float32Array.from({ length: vocabSize * hiddenSize }, (_, i) => ((i * 7) % 29 - 14) / 32);
  const hidden = acquireBuffer(input.byteLength, undefined, 'paired_logits_input');
  const norm = acquireBuffer(normValues.byteLength, undefined, 'paired_logits_norm');
  const head = acquireBuffer(headValues.byteLength, undefined, 'paired_logits_head');
  device.queue.writeBuffer(hidden, 0, input);
  device.queue.writeBuffer(norm, 0, normValues);
  device.queue.writeBuffer(head, 0, headValues);
  const weights = { finalNorm: createWeightBuffer(norm, 'f32', 'row', [hiddenSize]),
    lmHead: createWeightBuffer(head, 'f32', 'row', [vocabSize, hiddenSize]) };
  const config = { hiddenSize, vocabSize, rmsNormEps: 1e-6, rmsNormWeightOffset: false,
    normalizationType: 'rmsnorm', activationDtype: 'f32', activationPrecision: 'two-f32',
    activationComponents: 2, activationElementStrideBytes: 8, useTiedEmbeddings: false,
    finalLogitSoftcapping: null, logitInputScale: 1, logitOutputScale: 1 };
  // Independent Number/F64 equations from the exact frozen F32 components.
  const reference = [];
  for (let token = 0; token < 2; token++) {
    const values = Array.from({ length: hiddenSize }, (_, k) => input[(token * hiddenSize + k) * 2] + input[(token * hiddenSize + k) * 2 + 1]);
    const inverse = 1 / Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / hiddenSize + config.rmsNormEps);
    for (let n = 0; n < vocabSize; n++) reference.push(Math.fround(values.reduce((sum, value, k) =>
      sum + value * inverse * normValues[k] * headValues[n * hiddenSize + k], 0)));
  }
  const check = (values, expected) => {
    assert.equal(values.length, expected.length);
    for (let i = 0; i < values.length; i++) assert(Math.abs(values[i] - expected[i]) <= 1e-6,
      `logit ${i}: ${values[i]} versus ${expected[i]}`);
  };
  const outputs = {};
  try {
    const active = getBufferPool().getStats().activeBuffers;
    const snapshot = await snapshotTensor(hidden, [2, hiddenSize], 'f32', {
      precision: 'two-f32', components: 2, elementStrideBytes: 8, includeData: true,
    });
    assert.equal(snapshot.ok, true);
    assert.deepEqual(snapshot.storageComponents, Array.from(input));
    assert.deepEqual(snapshot.data, Array.from({ length: 2 * hiddenSize }, (_, i) => input[2 * i] + input[2 * i + 1]));
    const execute = (overrides, options) => computeLogits(hidden, 2, { ...weights, ...overrides }, config,
      true, {}, undefined, undefined, null, options);
    outputs.allRows = Array.from(await execute({}, null));
    check(outputs.allRows, reference);
    assert.equal(getBufferPool().getStats().activeBuffers, active);
    outputs.lastRow = Array.from(await execute({}, { lastPositionOnly: true }));
    check(outputs.lastRow, reference.slice(vocabSize));
    assert.equal(getBufferPool().getStats().activeBuffers, active);
    await assert.rejects(execute({ lmHeadBias: new Float32Array(1) }, null), /bias requires/);
    assert.equal(getBufferPool().getStats().activeBuffers, active, 'immediate failure settles paired temporaries');
    const recorder = new CommandRecorder(device, 'paired_logits_recorded');
    let output;
    try {
      output = await recordLogitsGPU(recorder, hidden, 2, weights, config);
      await recorder.submitAndWait();
      assert.equal(output.logitsDtype, 'f32');
      outputs.recorded = Array.from(new Float32Array(await readBuffer(output.logitsBuffer, 2 * vocabSize * 4)));
      check(outputs.recorded, reference);
    } finally { await recorder.abort(); if (output) releaseBuffer(output.logitsBuffer); }
    assert.equal(getBufferPool().getStats().activeBuffers, active);
    const failedRecorder = new CommandRecorder(device, 'paired_logits_rejected');
    try {
      await assert.rejects(recordLogitsGPU(failedRecorder, hidden, 2,
        { ...weights, lmHead: createWeightBuffer(head, 'bf16', 'row', [vocabSize, hiddenSize]) }, config), /F32\/F16\/Q4K/);
    } finally { await failedRecorder.abort(); }
    assert.equal(getBufferPool().getStats().activeBuffers, active, 'recorded projection failure settles prior norm resources');
    const direct = await computeLogitsGPU(hidden, 2, weights, config, {});
    try {
      assert.equal(direct.logitsDtype, 'f32');
      outputs.direct = Array.from(new Float32Array(await readBuffer(direct.logitsBuffer, 2 * vocabSize * 4)));
      check(outputs.direct, reference);
    } finally { releaseBuffer(direct.logitsBuffer); }
    assert.equal(getBufferPool().getStats().activeBuffers, active);
    await assert.rejects(computeLogitsGPU(hidden, 2,
      { ...weights, lmHeadBias: new Float32Array(1) }, config, {}), /bias requires/);
    assert.equal(getBufferPool().getStats().activeBuffers, active, 'direct finalization failure settles raw logits');
    const biasRecorder = new CommandRecorder(device, 'paired_logits_bias_rejected');
    try {
      await assert.rejects(recordLogitsGPU(biasRecorder, hidden, 2,
        { ...weights, lmHeadBias: new Float32Array(1) }, config), /bias requires/);
    } finally { await biasRecorder.abort(); }
    assert.equal(getBufferPool().getStats().activeBuffers, active, 'recorded finalization failure settles raw logits');
    console.log(JSON.stringify({ schema: 'doppler.paired-logits-control/v1', passed: true,
      checks: ['all-rows', 'last-row-copy', 'immediate-failure', 'recorded', 'recorded-failure',
        'direct', 'direct-finalization-failure', 'recorded-finalization-failure', 'paired-snapshot'],
      tolerance: 1e-6, input: Array.from(input), norm: Array.from(normValues), head: Array.from(headValues),
      reference, outputs, provider: probe.providerReceipt }));
  } finally { releaseBuffer(hidden); releaseBuffer(norm); releaseBuffer(head); }
} else {
  console.log(JSON.stringify({ schema: 'doppler.paired-logits-control/v1', passed: false, skipped: probe.reason }));
}
setGPUDevice(null);
destroyDevice();
