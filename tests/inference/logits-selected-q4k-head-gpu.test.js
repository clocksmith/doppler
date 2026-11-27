import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, getBufferPool, releaseBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { quantizeToQ4KMRowWise } from '../../src/converter/quantizer.js';
import { computeLogits } from '../../src/inference/pipelines/text/logits/index.js';

// scoreChoices reads selected-token logits. A q4k output head used to throw
// "[LmHeadSelect] selected-token LM head requires f16 weights"; it now takes the
// full q4k matmul and gathers the requested rows, which must equal the full logits.
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const hiddenSize = 256, vocabSize = 16;
  let seed = 7;
  const next = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 2 ** 32 - 0.5; };
  const headValues = Float32Array.from({ length: vocabSize * hiddenSize }, next);
  const hiddenValues = Float32Array.from({ length: hiddenSize }, next);
  const { quantized } = quantizeToQ4KMRowWise(headValues, [vocabSize, hiddenSize]);
  const hidden = acquireBuffer(hiddenValues.byteLength, undefined, 'selected_q4k_hidden');
  const norm = acquireBuffer(hiddenSize * 4, undefined, 'selected_q4k_norm');
  const head = acquireBuffer(quantized.byteLength, undefined, 'selected_q4k_head');
  uploadData(hidden, hiddenValues);
  uploadData(norm, new Float32Array(hiddenSize).fill(1));
  uploadData(head, quantized);
  const weights = { finalNorm: createWeightBuffer(norm, 'f32', 'row', [hiddenSize]),
    lmHead: createWeightBuffer(head, 'q4k', 'row', [vocabSize, hiddenSize]) };
  const config = { hiddenSize, vocabSize, rmsNormEps: 1e-6, rmsNormWeightOffset: false,
    normalizationType: 'rmsnorm', activationDtype: 'f32', useTiedEmbeddings: false,
    finalLogitSoftcapping: null, logitInputScale: 1, logitOutputScale: 1 };
  const run = (options) => computeLogits(hidden, 1, weights, config, true, {}, undefined, undefined, undefined,
    { lastPositionOnly: true, ...options });
  const selectedIds = [3, 0, 11];
  try {
    const before = getBufferPool().getStats().activeBuffers;
    const full = await run({});
    const selected = await run({ selectedTokenIds: selectedIds });
    assert.equal(selected.length, selectedIds.length);
    selectedIds.forEach((tokenId, index) => {
      assert.ok(Number.isFinite(selected[index]));
      assert.ok(Math.abs(selected[index] - full[tokenId]) < 1e-4,
        `selected logit for ${tokenId} must equal the full q4k logit (${selected[index]} vs ${full[tokenId]})`);
    });
    assert.equal(getBufferPool().getStats().activeBuffers, before);
  } finally {
    releaseBuffer(hidden); releaseBuffer(norm); releaseBuffer(head);
  }
}
destroyDevice();
console.log(`logits-selected-q4k-head-gpu: ${probe.ready ? 'GPU checks passed' : `skipped: ${probe.reason}`}`);
