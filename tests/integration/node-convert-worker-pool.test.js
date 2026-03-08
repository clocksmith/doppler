import assert from 'node:assert/strict';

import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { NodeConvertWorkerPool } = await import('../../src/tooling/node-convert-worker-pool.js');
const { transformTensorBytes, shouldQuantize } = await import('../../src/converter/core.js');
const { float32ToFloat16 } = await import('../../src/converter/quantizer.js');

function buildF16Data(length) {
  const out = new Uint16Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = float32ToFloat16(((i % 37) - 18) / 9);
  }
  return new Uint8Array(out.buffer.slice(0));
}

assert.throws(
  () => new NodeConvertWorkerPool({ size: 0 }),
  /positive integer/
);
assert.throws(
  () => new NodeConvertWorkerPool({ size: 1.5 }),
  /positive integer/
);

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
    layout: 'row',
    variantTag: 'q4k-ehaf16',
  },
  quantizeEmbeddings: false,
};

const direct = transformTensorBytes(tensor, sourceBytes, transformContext);
assert.equal(direct.outDtype, 'Q4_K_M');
assert.equal(direct.outLayout, 'row');

const pool = new NodeConvertWorkerPool({ size: 2 });
assert.equal(pool.size, 2);
try {
  const workerFull = await pool.transformTensor(tensor, sourceBytes.slice(), transformContext);
  assert.equal(workerFull.outDtype, direct.outDtype);
  assert.equal(workerFull.outLayout, direct.outLayout);
  assert.deepEqual(workerFull.tensorData, direct.tensorData);

  const padded = new Uint8Array(sourceBytes.byteLength + 2);
  padded.set(sourceBytes, 1);
  const offsetView = padded.subarray(1, 1 + sourceBytes.byteLength);
  const workerFromOffsetView = await pool.transformTensor(tensor, offsetView, transformContext);
  assert.deepEqual(workerFromOffsetView.tensorData, direct.tensorData);

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

  await assert.rejects(
    () => pool.transformTensor(null, sourceBytes, transformContext),
    /tensor is required/
  );
  await assert.rejects(
    () => pool.transformTensor(tensor, new ArrayBuffer(sourceBytes.byteLength), transformContext),
    /must be Uint8Array/
  );
  await assert.rejects(
    () => pool.transformTensor(
      {
        name: 'bad-f32',
        shape: [3],
        dtype: 'F32',
      },
      new Uint8Array([1, 2, 3]),
      { targetQuant: 'f16' }
    ),
    /Invalid F32 tensor byte length/
  );
  await assert.rejects(
    () => pool.transformTensor(
      {
        ...tensor,
        uncloneable: () => {},
      },
      sourceBytes,
      transformContext
    ),
    /clone|function|could not be cloned/i
  );
} finally {
  await pool.close();
}

const closePool = new NodeConvertWorkerPool({ size: 1 });
const closePending = [];
for (let i = 0; i < 12; i += 1) {
  const pending = closePool.transformTensor(tensor, sourceBytes.slice(), transformContext);
  pending.catch(() => {});
  closePending.push(pending);
}
await closePool.close();
const closeResults = await Promise.allSettled(closePending);
assert.ok(closeResults.some((result) => result.status === 'rejected'));
for (const result of closeResults) {
  if (result.status !== 'rejected') continue;
  assert.match(String(result.reason?.message ?? result.reason), /closed|worker pool/i);
}

await closePool.close();
await assert.rejects(
  () => closePool.transformTensor(tensor, sourceBytes, transformContext),
  /closed|worker pool/i
);

console.log('node-convert-worker-pool.test: ok');
