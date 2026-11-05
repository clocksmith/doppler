import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU } from '../../src/inference/pipelines/text/logits/index.js';
import { PleConvState, runPleLayer } from '../../src/inference/pipelines/text/ngram/ple-layer.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const device = getDevice();
  const buffers = [];
  const upload = (values, label) => {
    const data = Float32Array.from(values);
    const buffer = acquireBuffer(data.byteLength, undefined, label); buffers.push(buffer);
    device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  const projection = (values, shape, label) => createWeightBuffer(upload(values, label), 'f32', 'row', shape, label);
  const hiddenValues = [1, 2, 3, 4, 4, 3, 2, 1, 2, 0, 0, 0, 0, 0, 0, 2];
  const hidden = upload(hiddenValues, 'streams');
  const weights = { finalNorm: null,
    hyperConnectionMixer: { norm: createTensor(upload(new Array(8).fill(0), 'hc_norm'), 'f32', [8]),
      down: projection(new Array(16).fill(0), [2, 8], 'mix_down'),
      up: projection(new Array(16).fill(0), [8, 2], 'mix_up'), inject: null },
    lmHead: projection([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [4, 4], 'lm_head') };
  const postLayer = ['hc_mixer_down', 'hc_mixer_up', 'lm_head', 'lm_head_prefill'].map(op => ({ op, kernel: 'matmul_f32.wgsl', entry: 'main' }));
  const config = { hiddenSize: 4, vocabSize: 4, activationDtype: 'f32', executionPhase: 'prefill',
    normalizationType: 'rmsnorm', rmsNormEps: 1e-6, rmsNormWeightOffset: true,
    finalLogitSoftcapping: null, logitInputScale: 1, logitOutputScale: 1, useTiedEmbeddings: false,
    hyperConnections: { streamCount: 2, streamWidth: 4, projectionRank: 2, normEps: 1e-6 },
    kernelPath: { id: 'root-mixer-test', preLayer: [], decode: { steps: [] }, prefill: { steps: [] }, postLayer } };
  // Independent closed-form case: zero mixing projections yield sigmoid(0)=1/2.
  // The mixer averages the two separately normalized streams; no final RMSNorm follows.
  const expected = [];
  for (let token = 0; token < 2; token++) {
    const streams = [0, 1].map(stream => hiddenValues.slice(token * 8 + stream * 4, token * 8 + stream * 4 + 4));
    const rms = streams.map(row => Math.sqrt(row.reduce((sum, x) => sum + x * x, 0) / 4 + 1e-6));
    for (let column = 0; column < 4; column++) expected.push((streams[0][column] / rms[0] + streams[1][column] / rms[1]) / 4);
  }
  const check = (actual, reference) => {
    assert.equal(actual.length, reference.length);
    actual.forEach((value, index) => assert.ok(Math.abs(value - reference[index]) < 2e-6,
      `logit ${index}: ${value} versus ${reference[index]}`));
  };
  try {
    const before = getBufferPool().getStats().activeBuffers;
    check(await computeLogits(hidden, 2, weights, config, true), expected);
    check(await computeLogits(hidden, 2, weights, config, true, {}, undefined, undefined, undefined, { lastPositionOnly: true }), expected.slice(4));
    const gpu = await computeLogitsGPU(hidden, 2, weights, config, {});
    try { check(new Float32Array(await readBuffer(gpu.logitsBuffer, 32)), expected); } finally { releaseBuffer(gpu.logitsBuffer); }
    const recorder = new CommandRecorder(device, 'root_mixer_logits');
    try {
      const recorded = await recordLogitsGPU(recorder, hidden, 2, weights, config);
      try {
        await recorder.submitAndWait();
        check(new Float32Array(await readBuffer(recorded.logitsBuffer, 32)), expected);
      } finally { releaseBuffer(recorded.logitsBuffer); }
    } finally { await recorder.abort(); }
    assert.equal(getBufferPool().getStats().activeBuffers, before, 'every adapter releases its intermediate mixer tensors');
    await assert.rejects(computeLogits(hidden, 2, { ...weights, finalNorm: weights.hyperConnectionMixer.norm.buffer }, config, true), /in place of the final norm/);
    assert.equal(getBufferPool().getStats().activeBuffers, before, 'ambiguous normalization fails without leaks');
    const convState = new PleConvState({ channels: 8, stateLength: 9 });
    try {
      const pleWeights = { keyProj: null, valueProj: null,
        normKey: weights.hyperConnectionMixer.norm, normQuery: weights.hyperConnectionMixer.norm,
        normConv: createTensor(upload(new Array(7).fill(0), 'bad_conv_norm'), 'f32', [7]),
        conv: createTensor(upload(new Array(32).fill(0), 'conv_weight'), 'f32', [32]) };
      const rows = acquireBuffer(8, undefined, 'bf16_rows'); buffers.push(rows);
      device.queue.writeBuffer(rows, 0, new Uint16Array(4));
      const borrowed = getBufferPool().getStats().activeBuffers;
      await assert.rejects(runPleLayer(null, { streams: createTensor(hidden, 'f32', [1, 8]), rows,
        weights: pleWeights, state: convState, ngram: { embedDim: 4, normEps: 1e-6, convKernelSize: 4, convDilation: 3, convStateLength: 9 },
        hyper: config.hyperConnections, numTokens: 1, project: async (_role, _input, _weight, outFeatures) => {
          const buffer = acquireBuffer(outFeatures * 4); device.queue.writeBuffer(buffer, 0, new Float32Array(outFeatures));
          return createTensor(buffer, 'f32', [1, outFeatures]);
        } }), /norm weight/);
      assert.equal(getBufferPool().getStats().activeBuffers, borrowed, 'a failure after PLE gating releases gateInput too');
    } finally { await convState.destroy(); }
  } finally { for (const buffer of buffers) releaseBuffer(buffer); }
}
destroyDevice();
console.log(`hyper-connection logits: ${probe.ready ? 'physical GPU numerical and ownership checks passed' : `skipped: ${probe.reason}`}`);
