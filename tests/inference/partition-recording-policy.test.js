import assert from 'node:assert/strict';
import { shouldRecordPartitionLayers } from '../../src/inference/pipelines/text/partitions/partition-execution.js';

const state = {
  currentSeqLen: 0,
  runtimeConfig: { inference: { session: { prefillTokenChunkSize: null } } },
  modelConfig: { sessionSettings: { prefillTokenChunkSize: null } },
  kvCache: { layout: 'contiguous' },
  dopplerLoader: { getRoutedExpertPolicy: () => ({ enabled: true }) },
};
assert.equal(shouldRecordPartitionLayers(state, false, true, true), false,
  'Routed multi-token prefill must retain the immediate executor when decode recording is enabled');
assert.equal(shouldRecordPartitionLayers({ ...state, currentSeqLen: 36 }, false, false, true), true,
  'Ordinary single-token routed decode can use its dependency-segmented recorder');
for (const prefill of [false, true]) {
  assert.equal(shouldRecordPartitionLayers(state, true, prefill, true), false,
    'An explicit command-batching disable remains authoritative for both phases');
}
const dense = { ...state, dopplerLoader: { getRoutedExpertPolicy: () => ({ enabled: false }) } };
assert.equal(shouldRecordPartitionLayers(dense, false, true, true), true);
assert.equal(shouldRecordPartitionLayers({ ...dense, kvCache: { layout: 'bdpa_paged' } }, false, true, true), false);
const chunked = { ...dense, runtimeConfig: { inference: { session: { prefillTokenChunkSize: 16 } } } };
assert.equal(shouldRecordPartitionLayers(chunked, false, true, false), false);
assert.equal(shouldRecordPartitionLayers({ ...chunked, currentSeqLen: 16 }, false, true, true), false);
assert.throws(() => shouldRecordPartitionLayers(state, undefined, true, true), /resolved boolean/);
console.log('Partition recording preserves routed, paged, chunked and explicit-disable prefill policies while allowing segmented decode.');
