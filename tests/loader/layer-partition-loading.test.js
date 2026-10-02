import assert from 'node:assert/strict';
import { load } from '../../src/loader/model-executor.js';
import { createLayerPartitionPlan, resolveLayerPartition } from '../../src/inference/pipelines/text/layer-partition-contract.js';
import { createRDRRManifestFixture } from '../helpers/rdrr-manifest-fixture.js';

// Exercise the production load coordinator with observed materialization ports.
// This does not claim GPU execution or selective artifact acquisition.
function fixture({ tied = false, failLayer = null, head = null } = {}) {
  const manifest = createRDRRManifestFixture();
  manifest.architecture.numLayers = 4;
  manifest.inference.output.tieWordEmbeddings = tied;
  const calls = [];
  const progress = [];
  const loader = {
    manifest, calls, heapManager: {}, modelId: null, isLoaded: false,
    isUnifiedMemory: true, tensorLocations: new Map(), layers: new Map(),
    experts: new Map(), gpuBuffers: new Set(), _memoryMonitor: null,
    _loadingConfig: { memoryManagement: {
      flushIntervalLayers: 1, flushThresholdBytes: 1024, gpuQueueFlushLayers: 8,
    } },
    shardCache: {
      size: 0, totalBytes: 0, hasCustomLoader: true,
      customReadStats: { bytesRead: 0, shardsRead: 0 },
      configureForModel() {}, resetCustomReadStats() {}, clear() {},
    },
    _startMemoryLogging() { this._memoryMonitor = {}; },
    _stopMemoryLogging() { this._memoryMonitor = null; },
    _assertResidentBudget() {},
    async _buildTensorLocations() {
      if (head) this.tensorLocations.set('output', { role: 'lm_head', group: 'head', shape: [8, 4], ...head });
    },
    async _loadShard() { throw new Error('Unexpected shard acquisition in materialization-port test'); },
    async _loadEmbeddings() { calls.push('embed'); },
    async _loadFinalWeights() { calls.push('head'); },
    _prefetchLayerShards(layer) { calls.push(`prefetch:${layer}`); },
    async _loadLayer(layer) {
      calls.push(layer);
      this.layers.set(layer, {});
      if (layer === failLayer) throw new Error('materialization failed');
    },
    async unload() {
      calls.push('unload');
      this.layers.clear(); this.modelId = null; this.isLoaded = false;
      this.manifest = null;
    },
  };
  const plan = createLayerPartitionPlan({ modelId: manifest.modelId,
    numLayers: 4, hiddenSize: 4, vocabSize: 8, splitLayer: 2 });
  const execute = partition => load.call(loader, manifest.modelId, {
    verifyHashes: true, partition, onProgress: event => progress.push(event),
  });
  return { manifest, loader, plan, calls, progress, execute };
}

// Source tying does not make a separately quantized head depend on input
// embeddings. Dense aliases and absent heads still retain the shared weight.
for (const [head, shared] of [[{ dtype: 'Q4_K' }, false], [{ dtype: 'F16' }, true],
  [{ dtype: 'F16', storage: { encoding: 'q4k' } }, false], [null, true]]) {
  const f = fixture({ tied: true, head });
  await f.execute({ plan: f.plan, index: 1 });
  assert.deepEqual(f.calls, shared ? ['embed', 2, 3, 'head'] : [2, 3, 'head']);
}

for (const tied of [false, true]) {
  for (const index of [0, 1]) {
    const f = fixture({ tied });
    await f.execute({ plan: f.plan, index });
    assert.deepEqual(f.calls, index === 0 ? ['embed', 0, 1]
      : tied ? ['embed', 2, 3, 'head'] : [2, 3, 'head']);
    assert.deepEqual([...f.loader.layers.keys()], index === 0 ? [0, 1] : [2, 3]);
    assert.equal(f.loader.loadTiming.layers.count, 2);
    assert.deepEqual(f.progress.filter(p => p.stage === 'layers').map(p => [p.layer, p.total]), [[1, 2], [2, 2]]);
    assert.equal(f.loader.isLoaded, true);
    assert.equal(f.loader._memoryMonitor, null);
  }
}

{
  const f = fixture();
  await f.execute(null);
  assert.deepEqual(f.calls, ['embed', 0, 'prefetch:0', 1, 'prefetch:1', 2, 'prefetch:2', 3, 'prefetch:3', 'head']);
  assert.equal(f.loader.loadTiming.layers.count, 4);
}

{
  const f = fixture({ failLayer: 2 });
  await assert.rejects(f.execute({ plan: f.plan, index: 1 }), /materialization failed/);
  assert.deepEqual(f.calls, [2, 'unload']);
  assert.equal(f.loader.layers.size, 0);
  assert.equal(f.loader.isLoaded, false);
  assert.equal(f.loader._memoryMonitor, null);
  assert.equal(f.loader.manifest, f.manifest);
  assert.equal(f.loader.loadTiming.status, 'failed');
}

for (const mutate of [
  p => { p.modelId = 'other-model'; },
  p => { p.partitions[1].layerRange = [0, 3]; },
  p => { p.partitions[0].hasLmHead = true; },
  p => { p.hiddenSize = 8; },
  p => { delete p.splitLayer; },
]) {
  const f = fixture();
  const plan = structuredClone(f.plan);
  mutate(plan);
  await assert.rejects(f.execute({ plan, index: 1 }));
  assert.deepEqual(f.calls, ['unload']);
  assert.equal(f.loader.loadTiming.status, 'failed');
  assert.equal(f.loader._memoryMonitor, null);
  assert.equal(f.loader.manifest, f.manifest);
}

{
  const f = fixture();
  const reordered = Object.fromEntries(Object.entries(structuredClone(f.plan)).reverse());
  assert.deepEqual(resolveLayerPartition(f.manifest, { plan: reordered, index: 1 }).layerRange, [2, 3]);
  assert.throws(() => { f.plan.partitions[0].layerRange[1] = 3; }, TypeError);
  assert.throws(() => { f.plan.partitions[0].outputContract.dtype = 'f16'; }, TypeError);
  f.loader.heapManager = null;
  let resume;
  f.loader.init = () => new Promise(resolve => { resume = resolve; });
  const allocation = { plan: structuredClone(f.plan), index: 0 };
  const pending = f.execute(allocation);
  allocation.index = 1;
  allocation.plan.partitions[0].layerRange[1] = 3;
  resume();
  await pending;
  assert.deepEqual(f.calls, ['embed', 0, 1]);
}

console.log('layer-partition-loading.test: passed (materialization coordination only)');
