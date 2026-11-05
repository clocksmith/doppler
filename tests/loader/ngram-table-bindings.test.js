import assert from 'node:assert/strict';
import { bindNgramTables } from '../../src/loader/ngram-table-bindings.js';
import { NgramTableReader } from '../../src/inference/pipelines/text/ngram/ngram-table.js';
import { ShardCache } from '../../src/loader/shard-cache.js';

const name = index => `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_${index}.weight`;
const calls = [];
const source = Uint8Array.from({ length: 64 }, (_, i) => i);
const loader = { tensorLocations: new Map([
  [name(0), { shape: [2, 2], dtype: 'BF16', role: 'ngram_table', size: 8, shardIndex: 0, offset: 16 }],
  [name(1), { shape: [2, 2], dtype: 'BF16', role: 'ngram_table', size: 8, shardIndex: 0, offset: 40 }],
]), shardCache: { loadRange: async (index, start, count, options) => {
  calls.push({ index, start, count, options }); return source.slice(start, start + count).buffer;
} } };
const contract = { tableDtype: 'bf16', headDim: 2, layers: [{ decoderLayerIndex: 1, rows: 4 }] };
const session = { rowCache: { maxRows: 2, maxBytes: 8, eviction: 'lru' }, staging: { maxRows: 2 }, reads: { maxConcurrent: 1 } };
const options = bindNgramTables(loader, contract, session).get(1);
assert.equal(calls.length, 0, 'binding does not read the table');
const reader = new NgramTableReader(options);
assert.deepEqual([...await reader.gather([3, 0, 3])], [...source.slice(44, 48), ...source.slice(16, 20), ...source.slice(44, 48)]);
assert.ok(calls.every(call => call.count === 4 && call.options.requireRange === true));
assert.equal(reader.stats().rowBytesVerified, false);
const malformed = { ...loader, tensorLocations: new Map(loader.tensorLocations) };
malformed.tensorLocations.delete(name(0));
assert.throws(() => bindNgramTables(malformed, contract, session), /Missing or ambiguous|parts cover|contract declares/);
let fullLoads = 0;
const cache = new ShardCache({ maxEntries: 1, verifyHashes: false,
  loadingConfig: { verifyHashes: false, maxConcurrentLoads: 0, opfsEntries: 1, networkEntries: 1, moeMaxEntries: 1 },
  customLoader: async () => { fullLoads++; return new ArrayBuffer(64); },
  customRangeLoader: async () => { const error = new Error('Range unsupported'); error.code = 'ERR_RANGE_UNSUPPORTED'; throw error; },
});
await assert.rejects(cache.loadRange(0, 16, 4, { requireRange: true }), /bounded range|Range unsupported/);
assert.equal(fullLoads, 0, 'a table row read must never fall back to loading the full checkpoint file');
const delayed = Promise.withResolvers();
const started = Promise.withResolvers();
let settled = false;
const concurrent = new NgramTableReader({ ...options,
  session: { ...session, reads: { maxConcurrent: 2 } },
  readRange: async (_source, offset) => {
    if (offset === 16) throw Error('row read failed');
    started.resolve(); return delayed.promise;
  } });
const pending = concurrent.gather([0, 2]);
const rejected = assert.rejects(pending, /row read failed/).then(() => { settled = true; });
await started.promise;
await new Promise(resolve => setImmediate(resolve));
assert.equal(settled, false, 'failed batches retain the other in-flight reads until settlement');
delayed.resolve(new Uint8Array(4));
await rejected;
assert.equal(concurrent.stats().cachedRows, 0);
console.log('ngram-table-bindings: bounded row I/O and fail-closed whole-file fallback passed');
