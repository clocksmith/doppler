import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, getBufferPool, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { computeLogits } from '../../src/inference/pipelines/text/logits/index.js';
import { projectHiddenStates } from '../../src/inference/pipelines/text/generator/hidden-projection.js';

// The smallest pipeline state the projection reads: weights, model config, the
// active plan's activation dtype and the debug settings.
function stateFor(weights, overrides = {}) {
  return {
    weights,
    useGPU: true,
    debugFlags: {},
    useTiedEmbeddings: false,
    embeddingVocabSize: null,
    resolvedKernelPath: null,
    executionV1State: null,
    modelConfig: { hiddenSize: 4, vocabSize: 4, rmsNormEps: 1e-6, rmsNormWeightOffset: false, normalizationType: 'rmsnorm',
      finalLogitSoftcapping: null, logitInputScale: 1, logitOutputScale: 1, ...overrides.modelConfig },
    executionPlanState: { primaryPlan: { id: 'primary', activationDtype: 'f32', kernelPath: null }, fallbackPlan: null, activePlanId: 'primary' },
    runtimeConfig: { inference: { largeWeights: null, compute: { activationDtype: 'f32' } }, shared: { debug: { probes: [] } } },
    ...overrides.state,
  };
}

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const device = getDevice();
  const norm = acquireBuffer(16, undefined, 'projection_norm');
  const head = acquireBuffer(32, undefined, 'projection_head');
  device.queue.writeBuffer(norm, 0, new Float32Array(4).fill(1));
  // Exact binary16 identity: each logit row equals its normalized hidden state.
  device.queue.writeBuffer(head, 0, Uint16Array.from([0x3c00, 0, 0, 0, 0, 0x3c00, 0, 0, 0, 0, 0x3c00, 0, 0, 0, 0, 0x3c00]));
  const weights = new Map([
    ['final_norm', createWeightBuffer(norm, 'f32', 'row', [4])],
    ['lm_head', createWeightBuffer(head, 'f16', 'row', [4, 4])],
  ]);
  try {
    let state = stateFor(weights);
    // Find out whether this minimal state resolves an execution plan; the
    // projection must use the plan the pipeline itself would.
    const before = getBufferPool().getStats().activeBuffers;
    const rows = Float32Array.from([1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0, 2]);
    const { logits, rowCount, vocabSize } = await projectHiddenStates(state, rows, 3);
    assert.deepEqual([rowCount, vocabSize, logits.length], [3, 4, 12]);
    const scale = [Math.sqrt(7.5 + 1e-6), Math.sqrt(7.5 + 1e-6), Math.sqrt(1 + 1e-6)];
    for (let i = 0; i < 12; i++) assert.ok(Math.abs(logits[i] - rows[i] / scale[Math.floor(i / 4)]) < 1e-5, `logit ${i}`);
    // Several rows at once agree with the runtime's own one-row logits for each row.
    for (let row = 0; row < 3; row++) {
      const single = await computeLogits(rows.slice(row * 4, row * 4 + 4), 1, { finalNorm: weights.get('final_norm'), lmHead: weights.get('lm_head') },
        { hiddenSize: 4, vocabSize: 4, rmsNormEps: 1e-6, rmsNormWeightOffset: false, normalizationType: 'rmsnorm', activationDtype: 'f32',
          useTiedEmbeddings: false, finalLogitSoftcapping: null, logitInputScale: 1, logitOutputScale: 1 }, true, {});
      for (let i = 0; i < 4; i++) assert.ok(Math.abs(single[i] - logits[row * 4 + i]) < 1e-6);
    }
    assert.equal(getBufferPool().getStats().activeBuffers, before, 'The projection releases every buffer it acquired');

    await assert.rejects(projectHiddenStates(state, rows, 2), /expected 2 rows of 4 values \(8\), got 12/);
    await assert.rejects(projectHiddenStates(state, [1, 2, 3, 4], 1), /Float32Array/);
    await assert.rejects(projectHiddenStates(state, Float32Array.from([1, NaN, 3, 4]), 1), /non-finite value at index 1/);
    state = stateFor(weights, { modelConfig: { hyperConnections: { streams: 2 } } });
    await assert.rejects(projectHiddenStates(state, rows, 3), /hyper-connection/);
    assert.equal(getBufferPool().getStats().activeBuffers, before);
  } finally {
    releaseBuffer(norm); releaseBuffer(head);
  }
}
destroyDevice();
console.log(`hidden-projection-gpu: ${probe.ready ? 'GPU checks passed' : `skipped: ${probe.reason}`}`);
