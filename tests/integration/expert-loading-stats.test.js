import assert from 'node:assert/strict';
import { InferencePipeline } from '../../src/inference/pipelines/text/pipeline.js';
import { snapshotExpertLoadingStats } from '../../src/inference/pipelines/text/expert-loading-stats.js';
import { createModelHandle } from '../../src/client/model-host/model-session.js';

const counters = { cacheHits: 2, phaseProfiles: {}, compactLayers: { 0: { cacheMisses: 3 } } };
const readback = { scope: 'Includes queued GPU wait; not GPU compute.', phases: {} };
let cacheReads = 0, readbackReads = 0;
const owner = { isLoaded: true, runtimeConfig: { shared: { debug: { profiler: { enabled: false } } } },
  dopplerLoader: { getRoutedExpertCacheStats() { cacheReads++; return counters; },
    getRoutedReadbackStats() { readbackReads++; return readback; } } };

const first = snapshotExpertLoadingStats(owner);
assert.equal(first.schema, 'doppler.expert-loading-stats/v1');
assert.equal(first.profilerEnabled, false);
assert.deepEqual(first.routedExpertCache.phaseProfiles, {});
assert.deepEqual(first.routedReadback.phases, {});
first.routedExpertCache.compactLayers[0].cacheMisses = 999;
first.routedReadback.phases.decode = { calls: 9 };
assert.equal(counters.compactLayers[0].cacheMisses, 3);
assert.deepEqual(readback.phases, {});
owner.runtimeConfig.shared.debug.profiler.enabled = true;
counters.phaseProfiles.decode = { preparations: 1, counters: { sourceReads: 3 } };
readback.phases.decode = { calls: 1, bytes: 40, wallMs: 0.3 };
const second = InferencePipeline.prototype.getExpertLoadingStats.call(owner);
assert.equal(second.profilerEnabled, true);
assert.equal(second.routedExpertCache.phaseProfiles.decode.counters.sourceReads, 3);
counters.phaseProfiles.decode.counters.sourceReads = 6;
assert.equal(second.routedExpertCache.phaseProfiles.decode.counters.sourceReads, 3);
assert.throws(() => snapshotExpertLoadingStats({ ...owner, isLoaded: false }), /loaded text pipeline/);
assert.throws(() => snapshotExpertLoadingStats({ ...owner, isGenerating: true }), /idle model boundary/);
assert.throws(() => snapshotExpertLoadingStats({ isLoaded: true }), /does not support/);

const pipeline = { ...owner, manifest: {}, getStats() { return { ordinary: true }; },
  getExpertLoadingStats() { return snapshotExpertLoadingStats(this); },
  async unload() { this.isLoaded = false; } };
const model = createModelHandle(pipeline, { modelId: 'counter-fixture' });
const beforeOrdinary = [cacheReads, readbackReads];
assert.deepEqual(model.advanced.getStats(), { ordinary: true });
assert.deepEqual([cacheReads, readbackReads], beforeOrdinary, 'Ordinary stats never capture loader snapshots');
assert.equal(model.advanced.getExpertLoadingStats().routedExpertCache.cacheHits, 2);
await model.unload();
assert.throws(() => model.advanced.getExpertLoadingStats(), /open loaded model/);
const unsupported = createModelHandle({ isLoaded: true, manifest: {} }, { modelId: 'unsupported' });
assert.throws(() => unsupported.advanced.getExpertLoadingStats(), /does not support/);
console.log('Expert loading snapshots: explicit detached CPU counters, ordinary stats unchanged, profiler visibility and handle lifetime passed');
