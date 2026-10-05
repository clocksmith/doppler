import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/inference/pipelines/text/partition-attempt.js', import.meta.url).href;
const allocations = [];
globalThis.partitionCacheAllocation = (config, range) => {
  const cache = { maxSeqLen: config.session.kvcache.maxSeqLen, destroyed: false,
    destroy() { this.destroyed = true; } };
  allocations.push({ config, range, cache }); return cache;
};
const modules = {
  './init.js': 'export const createKVCache = (_model, _gpu, _debug, config, range) => partitionCacheAllocation(config, range);',
  './partition-execution.js': 'export const assertPartitionExecutionSupported = () => {};',
  './layer-partition-contract.js': 'export const resolveLayerPartition = () => ({layerRange:[12,23]});',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return {
    url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true,
  };
  return next(specifier, context);
} });
const { createPartitionAttempt } = await import(target);
const kvcache = Object.freeze({ maxSeqLen: 6144, kvDtype: 'f16', layout: 'contiguous' });
const owner = { manifest: {}, modelConfig: { maxSeqLen: 262144 }, modelPartition: { plan: {} },
  runtimeConfig: { inference: { session: { kvcache } } }, executionPlanState: {}, weights: new Map() };
try {
  const short = createPartitionAttempt(owner, 24 + 4096);
  const retained = createPartitionAttempt(owner, 1588 + 4096);
  assert.equal(short.state.kvCache.maxSeqLen, 4120, 'Reserve prompt plus output allowance, not unused session capacity');
  assert.equal(retained.state.kvCache.maxSeqLen, 5684, 'The entire retained prompt and output allowance remain available');
  assert.equal(kvcache.maxSeqLen, 6144, 'The prepared session policy remains immutable');
  for (const { config, range } of allocations) {
    assert.equal(config.session.kvcache.kvDtype, 'f16');
    assert.equal(config.session.kvcache.layout, 'contiguous');
    assert.deepEqual(range, [12, 23]);
  }
  short.close(); short.close();
  assert.equal(short.state.kvCache, null);
  assert.equal(retained.state.kvCache.destroyed, false, 'Closing one conversation preserves the other');
  retained.close();
  for (const invalid of [undefined, 0, -1, 6145]) {
    assert.throws(() => createPartitionAttempt(owner, invalid), /sequence allocation/);
  }
  for (const invalid of [undefined, null, {}, { maxSeqLen: undefined }]) {
    assert.throws(() => createPartitionAttempt({ ...owner, runtimeConfig: {
      inference: { session: { kvcache: invalid } },
    } }, 4120), /sequence allocation/);
  }
  assert.equal(allocations.length, 2, 'Reject invalid sequence allocations before acquiring cache resources');
} finally { hooks.deregister(); delete globalThis.partitionCacheAllocation; }
console.log('partition-attempt-allocation.test: bounded cache, immutable policy and isolated cleanup passed');
