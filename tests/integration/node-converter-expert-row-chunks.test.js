import assert from 'node:assert/strict';
import { createNodeTensorTransformer, createNodeLargeTensorTransformer, resolveRowChunkTransformPlan } from '../../src/tooling/node-converter/input.js';
import { transformTensorBytes, resolveTensorTargetQuant, normalizeStorageQuant, shouldQuantize } from '../../src/converter/tensor-transform.js';
import { NodeConvertWorkerPool } from '../../src/tooling/node-convert-worker-pool.js';

const tensor = { name: 'model.layers.0.mlp.experts.down_proj', role: 'expert', dtype: 'F32', shape: [2, 3, 257] };
const values = Float32Array.from({ length: 2 * 3 * 257 }, (_, i) => Math.sin(i * 0.31) * (1 + Math.floor(i / 257)));
const bytes = new Uint8Array(values.buffer);
const execution = { effectiveWorkers: 2, rowChunkMinTensorBytes: 1, rowChunkRows: 2, maxInFlightJobs: 2 };
const transformContext = { targetQuant: 'q4k', q4kLayout: 'row', quantizeEmbeddings: false, modulesToNotConvert: null };
const helpers = { execution, resolveTensorTargetQuant, normalizeStorageQuant, shouldQuantize };
const direct = transformTensorBytes(tensor, bytes, transformContext);
const pool = new NodeConvertWorkerPool({ size: 2 });
try {
  const transform = createNodeTensorTransformer({ ...helpers, pool, transformTensorBytes });
  const result = await transform({ tensor, tensorData: bytes, transformContext });
  const writes = [];
  const streaming = createNodeLargeTensorTransformer({ ...helpers, pool,
    readRange: async (_path, offset, length) => bytes.slice(offset, offset + length) });
  const streamed = await streaming({ tensor: { ...tensor, size: bytes.byteLength, offset: 0, sourcePath: 'fixture' },
    transformContext, writeChunk: async value => writes.push(value.tensorData) });
  assert.equal(streamed.outDtype, 'Q4_K_M');
  assert.deepEqual(Buffer.concat(writes), Buffer.from(direct.tensorData), 'streamed bank must preserve expert/row order');
  assert.equal(result.outDtype, 'Q4_K_M');
  assert.equal(result.outLayout, 'row');
  assert.deepEqual(result.tensorData, direct.tensorData, 'parallel expert chunks must retain exact Q4K bytes, including partial row blocks');
  const plan = resolveRowChunkTransformPlan({ ...helpers, tensor, tensorByteLength: bytes.byteLength, transformContext });
  assert.equal(plan.rows, 6);
  assert.equal(plan.cols, 257);
  assert.equal(plan.forceQuantizeDecision, true);
  for (const rejected of [{ ...tensor, role: 'matmul' }, { ...tensor, shape: [2, 3, 0] }]) {
    assert.equal(resolveRowChunkTransformPlan({ ...helpers, tensor: rejected, tensorByteLength: bytes.byteLength, transformContext }), null);
  }
  assert.equal(resolveRowChunkTransformPlan({ ...helpers, tensor, tensorByteLength: bytes.byteLength,
    transformContext: { ...transformContext, q4kLayout: 'col' } }), null);
} finally { await pool.close(); }
console.log('node-converter-expert-row-chunks.test: ok');
