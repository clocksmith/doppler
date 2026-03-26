import assert from 'node:assert/strict';

import {
  buildTieredQuantAttentionOptions,
  buildContiguousQuantAttentionOptions,
} from '../../src/inference/pipelines/text/attention/quant-options.js';

const buffers = {
  rotation: { label: 'rotation' },
  codebook: { label: 'codebook' },
  residualK: { label: 'residual-k' },
  residualV: { label: 'residual-v' },
  residualNormsK: { label: 'residual-norms-k' },
  residualNormsV: { label: 'residual-norms-v' },
  qjl: { label: 'qjl' },
};

const prodKvState = {
  coldLen: 384,
  hotLen: 256,
  hotWindow: 256,
  hotStart: 128,
  coldPackedStride: 16,
  coldQuantMode: 'turboquant_prod',
  prodMode: true,
  residualPackedStride: 8,
  rotationMatrixBuffer: buffers.rotation,
  codebookCentroidsBuffer: buffers.codebook,
  residualKGPU: buffers.residualK,
  residualVGPU: buffers.residualV,
  residualNormsKGPU: buffers.residualNormsK,
  residualNormsVGPU: buffers.residualNormsV,
  qjlMatrixBuffer: buffers.qjl,
};

{
  const tiered = buildTieredQuantAttentionOptions(prodKvState, {
    seqLen: 1,
    numKVHeads: 8,
    causal: true,
    startPos: 512,
    slidingWindow: 0,
    attnSoftcap: 0,
    scale: 0.125,
  });

  assert.equal(tiered.mode, 'turboquant_prod');
  assert.equal(tiered.rotationMatrixBuffer, buffers.rotation);
  assert.equal(tiered.codebookCentroidsBuffer, buffers.codebook);
  assert.equal(tiered.residualKBuffer, buffers.residualK);
  assert.equal(tiered.residualNormsVBuffer, buffers.residualNormsV);
  assert.equal(tiered.qjlMatrixBuffer, buffers.qjl);
}

{
  const contiguous = buildContiguousQuantAttentionOptions(prodKvState, {
    seqLen: 1,
    kvLen: 640,
    numKVHeads: 8,
    causal: true,
    startPos: 639,
    slidingWindow: 0,
    attnSoftcap: 0,
    scale: 0.125,
  });

  assert.equal(contiguous.mode, 'turboquant_prod');
  assert.equal(contiguous.rotationMatrixBuffer, buffers.rotation);
  assert.equal(contiguous.codebookCentroidsBuffer, buffers.codebook);
  assert.equal(contiguous.residualVBuffer, buffers.residualV);
  assert.equal(contiguous.qjlMatrixBuffer, buffers.qjl);
  assert.equal(contiguous.packedStride, 16);
  assert.equal(contiguous.packedStrideMSE, 16);
  assert.equal(contiguous.packedStrideResidual, 8);
}

{
  const mseOnly = buildTieredQuantAttentionOptions({
    coldQuantMode: 'turboquant',
    coldPackedStride: 12,
    rotationMatrixBuffer: buffers.rotation,
    codebookCentroidsBuffer: buffers.codebook,
  }, {
    seqLen: 1,
    numKVHeads: 4,
    causal: true,
    startPos: 0,
    slidingWindow: 0,
    attnSoftcap: 0,
    scale: 0.25,
  });

  assert.equal(mseOnly.residualKBuffer, null);
  assert.equal(mseOnly.qjlMatrixBuffer, null);
}

console.log('attention-quant-options.test: ok');
