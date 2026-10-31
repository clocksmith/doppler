import assert from 'node:assert/strict';
import { load } from '../../src/loader/model-load.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { createLayerPartitionPlan } from '../../src/config/layer-partition-plan.js';
import { createRDRRManifestFixture } from '../helpers/rdrr-manifest-fixture.js';
import { setDevice } from '../../src/gpu/device.js';

// Exercise the production coordinator through materialization ports, without GPU claims.
function fixture({ mode = 'eager-resident', format = 'qwen3_5', materialize, dense = false } = {}) {
  const manifest = createRDRRManifestFixture();
  manifest.architecture.numLayers = 4;
  if (!dense) manifest.moeConfig = { numExperts: 8, numExpertsPerToken: 2,
    expertFormat: format, expertIntermediateSize: 8, sharedExpertIntermediateSize: 8 };
  const calls = [], progress = [], banks = new Map();
  const loader = {
    manifest, heapManager: {}, modelId: null, isLoaded: false, isUnifiedMemory: true,
    tensorLocations: new Map(), layers: new Map(), experts: new Map(), gpuBuffers: new Set(),
    _memoryMonitor: null,
    _loadingConfig: resolveRuntimeConfig({ loading: { expertPreparation: { mode } } }).loading,
    shardCache: { size: 0, totalBytes: 0, hasCustomLoader: true,
      customReadStats: { bytesRead: 0, shardsRead: 0 },
      configureForModel() {}, resetCustomReadStats() {}, clear() {} },
    _startMemoryLogging() { this._memoryMonitor = {}; },
    _stopMemoryLogging() { this._memoryMonitor = null; },
    _assertResidentBudget(stage) { calls.push(`budget:${stage}`); },
    async _buildTensorLocations() {},
    async _loadShard() { throw new Error('Unexpected shard request'); },
    async _loadEmbeddings() { calls.push('embeddings'); },
    async _loadFinalWeights() { calls.push('head'); },
    _prefetchLayerShards() {},
    async _loadLayer(layer) { calls.push(`layer:${layer}`); this.layers.set(layer, {}); },
    async loadExpert(layer, expert) {
      calls.push(`expert:${layer}:${expert}`);
      assert.equal(this.isLoaded, false, 'No ready flag while an expert bank is still loading');
      await materialize?.(layer);
      const bank = { numExperts: 8, expertFormat: format };
      banks.set(layer, bank);
      this.experts.set(`bank:${layer}`, bank);
      return bank;
    },
    async unload() {
      calls.push('unload'); this.isLoaded = false; this.modelId = null;
      this.layers.clear(); this.experts.clear(); banks.clear();
      this.manifest = null; this.loadTiming = null;
    },
  };
  const execute = options => load.call(loader, manifest.modelId, { verifyHashes: true,
    onProgress: event => progress.push(event), ...options });
  return { loader, manifest, calls, progress, banks, execute };
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

{
  const config = resolveRuntimeConfig({ loading: { expertPreparation: { mode: 'eager-resident' } } });
  assert.equal(config.loading.expertPreparation.mode, 'eager-resident');
  assert.equal(resolveRuntimeConfig().loading.expertPreparation.mode, 'on-demand');
  for (const expertPreparation of [null, 'eager-resident', { mode: 'eager' }, { mode: null }]) {
    assert.throws(() => resolveRuntimeConfig({ loading: { expertPreparation } }), /expertPreparation.mode/);
  }
}

for (const mode of ['on-demand', 'eager-resident']) {
  const f = fixture({ mode });
  await f.execute();
  assert.equal(f.loader.isLoaded, true);
  const receipt = f.loader.loadTiming.expertPreparation;
  assert.equal(receipt.mode, mode);
  assert.equal(receipt.status, mode === 'on-demand' ? 'deferred' : 'complete');
  assert.equal(receipt.layerCount, 4);
  assert.equal(receipt.preparedBanks, mode === 'on-demand' ? 0 : 4);
  assert.deepEqual(f.calls.filter(call => call.startsWith('expert:')),
    mode === 'on-demand' ? [] : [0, 1, 2, 3].map(layer => `expert:${layer}:0`));
  assert.equal(f.progress.at(-1).stage, 'complete');
  if (mode === 'eager-resident') {
    assert(f.calls.indexOf('head') < f.calls.indexOf('expert:0:0'));
    assert.equal(f.progress.filter(event => event.stage === 'experts').length, 4);
    assert(f.loader.loadTiming.phasesMs.expertPreparation >= 0);
  }
}

{
  const f = fixture({ dense: true });
  await f.execute();
  assert.equal(f.loader.loadTiming.expertPreparation, null, 'Dense models have no expert preparation');
  assert.equal(f.banks.size, 0);
}

{
  const f = fixture({ format: 'gemma4' });
  await f.execute();
  assert.equal(f.loader.loadTiming.expertPreparation.preparedBanks, 4);
}

{
  const f = fixture();
  const plan = createLayerPartitionPlan({ modelId: f.manifest.modelId,
    numLayers: 4, hiddenSize: 4, vocabSize: 8, splitLayer: 2 });
  await f.execute({ partition: { plan, index: 1 } });
  assert.deepEqual([...f.banks.keys()], [2, 3], 'Preparation respects assigned layer indices');
}

for (const format of ['mixtral', 'gpt-oss', 'unknown']) {
  const f = fixture({ format });
  await assert.rejects(f.execute(), /unsupported for expertFormat/);
  assert.equal(f.loader.layers.size, 0, 'Reject unsupported eager residency before base-weight allocation');
  assert(!f.calls.some(call => call.startsWith('expert:')));
}

{
  const f = fixture();
  delete f.loader._loadingConfig.expertPreparation;
  await assert.rejects(f.execute(), /expertPreparation.mode/, 'Direct callers must supply resolved policy');
}

{
  const entered = deferred(), resume = deferred();
  const f = fixture({ async materialize(layer) {
    if (layer === 1) { entered.resolve(); await resume.promise; }
  } });
  const loading = f.execute();
  await entered.promise;
  assert.equal(f.loader.isLoaded, false);
  assert.equal(f.loader.loadTiming.expertPreparation.status, 'running');
  assert(!f.progress.some(event => event.stage === 'complete'));
  resume.resolve(); await loading;
  assert.equal(f.loader.isLoaded, true);
}

{
  const entered = deferred(), resume = deferred(), controller = new AbortController();
  const reason = new Error('Stop expert preparation');
  const f = fixture({ async materialize(layer) {
    if (layer === 1) { entered.resolve(); await resume.promise; }
  } });
  const loading = f.execute({ signal: controller.signal }).catch(error => error);
  await entered.promise;
  controller.abort(reason);
  await Promise.resolve();
  assert(!f.calls.includes('unload'), 'Cancellation cannot clean up an in-flight materialization');
  resume.resolve();
  assert.equal(await loading, reason);
  assert(f.calls.includes('unload'));
  assert(!f.calls.includes('expert:2:0'));
  assert.equal(f.loader.isLoaded, false);
  assert.equal(f.loader.experts.size, 0);
  assert.equal(f.loader._memoryMonitor, null);
  assert.equal(f.loader.loadTiming.status, 'failed');
  assert.equal(f.loader.loadTiming.expertPreparation.status, 'cancelled');
  assert.equal(f.loader.manifest, f.manifest, 'Custom source remains available for an explicit retry');
  assert(!f.progress.some(event => event.stage === 'complete'));
  await f.execute();
  assert.equal(f.loader.isLoaded, true, 'A settled cancelled load can be retried');
}

{
  const reason = new Error('Expert upload failed');
  const f = fixture({ async materialize(layer) { if (layer === 1) throw reason; } });
  await assert.rejects(f.execute(), error => error === reason);
  assert.equal(f.loader.loadTiming.expertPreparation.status, 'failed');
  assert.equal(f.loader.loadTiming.failedPhase, 'expertPreparation');
  assert(f.loader.loadTiming.phasesMs.expertPreparation >= 0);
  assert.equal(f.loader.isLoaded, false);
  assert.equal(f.banks.size, 0);
}

{
  const queued = deferred(), settled = deferred(), controller = new AbortController();
  setDevice({ features: new Set(), limits: {}, createBindGroup() {}, lost: new Promise(() => {}),
    queue: { submit() {}, onSubmittedWorkDone() { queued.resolve(); return settled.promise; } } }, { platformConfig: null });
  try {
    const f = fixture();
    const loading = f.execute({ signal: controller.signal }).catch(error => error);
    await queued.promise;
    assert.equal(f.banks.size, 4);
    assert.equal(f.loader.isLoaded, false, 'Uploads must settle before readiness');
    controller.abort();
    await Promise.resolve();
    assert(!f.calls.includes('unload'), 'Queued uploads retain ownership after cancellation');
    settled.resolve();
    assert.equal((await loading).name, 'AbortError');
    assert(f.calls.includes('unload'));
    assert.equal(f.loader.loadTiming.expertPreparation.status, 'cancelled');
  } finally { setDevice(null, { platformConfig: null }); }
}

{
  const controller = new AbortController(); controller.abort();
  const f = fixture();
  await assert.rejects(f.execute({ signal: controller.signal }), { name: 'AbortError' });
  assert.deepEqual(f.calls, [], 'Pre-aborted opening must not destroy an existing owner');
}

console.log('expert-preparation.test: preparation, deferred readiness, cancellation, cleanup and retry passed');
