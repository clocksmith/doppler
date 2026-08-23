import assert from 'node:assert/strict';
import {
  createRowChunks,
  mapOrderedChunkBatches,
} from '../../src/tooling/node-converter-chunk-batches.js';

const source = new Uint8Array(Array.from({ length: 24 }, (_, index) => index));
let activeJobs = 0;
let maxActiveJobs = 0;
const writes = [];
const chunks = createRowChunks({ rows: 6, rowChunkRows: 1, rowSourceBytes: 4 });
await mapOrderedChunkBatches({
  chunks,
  batchSize: 3,
  async transform(chunk) {
    activeJobs += 1;
    maxActiveJobs = Math.max(maxActiveJobs, activeJobs);
    await new Promise((resolve) => setImmediate(resolve));
    activeJobs -= 1;
    return source.slice(chunk.offset, chunk.offset + chunk.length);
  },
  async consume(bytes) {
    writes.push(...bytes);
  },
});

assert.equal(maxActiveJobs, 3, 'streamed row chunks must use bounded worker concurrency');
assert.deepEqual(writes, [...source], 'concurrent transforms must preserve source row order');

console.log('node-converter-large-tensor-batching.test: ok');
