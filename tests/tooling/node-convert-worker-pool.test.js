import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { NodeConvertWorkerPool } = await import('../../src/tooling/node-convert-worker-pool.js');
const { transformTensorBytes, shouldQuantize } = await import('../../src/converter/core.js');
const { float32ToFloat16 } = await import('../../src/converter/quantizer.js');

function buildF16Data(length) {
  const out = new Uint16Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = float32ToFloat16(((i % 37) - 18) / 9);
  }
  return new Uint8Array(out.buffer.slice(0));
}

const tensor = {
  name: 'model.layers.0.self_attn.q_proj.weight',
  shape: [4, 256],
  dtype: 'F16',
};

const sourceBytes = buildF16Data(tensor.shape[0] * tensor.shape[1]);
const transformContext = {
  targetQuant: 'q4k',
  q4kLayout: 'row',
  quantizationInfo: {
    weights: 'q4k',
    embeddings: 'f16',
    lmHead: 'f16',
    compute: 'f16',
    perLayerEmbeddings: 'int4_per_row',
    layout: 'row',
    variantTag: 'q4k-ehaf16',
  },
  quantizeEmbeddings: false,
};

const direct = transformTensorBytes(tensor, sourceBytes, transformContext);
assert.equal(direct.outDtype, 'Q4_K_M');
assert.equal(direct.outLayout, 'row');

const pool = new NodeConvertWorkerPool({ size: 2 });
try {
  const workerFull = await pool.transformTensor(tensor, sourceBytes.slice(), transformContext);
  assert.equal(workerFull.outDtype, direct.outDtype);
  assert.equal(workerFull.outLayout, direct.outLayout);
  assert.deepEqual(workerFull.tensorData, direct.tensorData);

  const rowBytes = tensor.shape[1] * 2;
  const forceQuantizeDecision = shouldQuantize(tensor.name, tensor.shape, {
    quantizeEmbeddings: false,
  });
  const chunkOutputs = [];
  for (let rowStart = 0; rowStart < tensor.shape[0]; rowStart += 2) {
    const rowCount = Math.min(2, tensor.shape[0] - rowStart);
    const start = rowStart * rowBytes;
    const end = start + (rowCount * rowBytes);
    const chunkTensor = {
      ...tensor,
      shape: [rowCount, tensor.shape[1]],
    };
    const chunkBytes = sourceBytes.slice(start, end);
    const workerChunk = await pool.transformTensor(chunkTensor, chunkBytes, {
      ...transformContext,
      forceQuantizeDecision,
    });
    chunkOutputs.push(workerChunk);
  }

  const combinedSize = chunkOutputs.reduce((sum, chunk) => sum + chunk.tensorData.byteLength, 0);
  const combined = new Uint8Array(combinedSize);
  let offset = 0;
  for (const chunk of chunkOutputs) {
    combined.set(chunk.tensorData, offset);
    offset += chunk.tensorData.byteLength;
  }
  assert.deepEqual(combined, direct.tensorData);

  const pleTensor = {
    name: 'model.language_model.embed_tokens_per_layer.weight',
    shape: [8, 4],
    dtype: 'F16',
  };
  const pleSourceBytes = buildF16Data(pleTensor.shape[0] * pleTensor.shape[1]);
  const pleDirect = transformTensorBytes(pleTensor, pleSourceBytes, transformContext);
  assert.ok(pleDirect.companionData instanceof Uint8Array);
  assert.ok(pleDirect.sourceTransform);

  const pleWorker = await pool.transformTensor(pleTensor, pleSourceBytes.slice(), transformContext);
  assert.deepEqual(pleWorker.tensorData, pleDirect.tensorData);
  assert.equal(pleWorker.outDtype, pleDirect.outDtype);
  assert.equal(pleWorker.outLayout, pleDirect.outLayout);
  assert.deepEqual(pleWorker.companionData, pleDirect.companionData);
  assert.deepEqual(pleWorker.sourceTransform, pleDirect.sourceTransform);

  const pleChunkTensor = {
    ...pleTensor,
    shape: [2, pleTensor.shape[1]],
  };
  const pleChunkBytes = pleSourceBytes.slice(0, 2 * pleTensor.shape[1] * 2);
  const pleChunkDirect = transformTensorBytes(pleChunkTensor, pleChunkBytes, {
    ...transformContext,
    originalTensorShape: pleTensor.shape,
  });
  const pleChunkWorker = await pool.transformTensor(pleChunkTensor, pleChunkBytes.slice(), {
    ...transformContext,
    originalTensorShape: pleTensor.shape,
  });
  assert.deepEqual(pleChunkWorker.tensorData, pleChunkDirect.tensorData);
  assert.deepEqual(pleChunkWorker.companionData, pleChunkDirect.companionData);
  assert.deepEqual(pleChunkWorker.sourceTransform, pleChunkDirect.sourceTransform);
  assert.deepEqual(pleChunkWorker.sourceTransform.storageShape, pleTensor.shape);
} finally {
  await pool.close();
}

console.log('node-convert-worker-pool.test: ok');
