import assert from 'node:assert/strict';
import { createInitializedPipeline } from '../../src/inference/pipelines/factory.js';
import { _loadWeights } from '../../src/inference/pipelines/text/lifecycle.js';

let constructed = 0;
const events = [];
class Pipeline {
  constructor() { constructed++; }
  async initialize() { events.push('initialize'); }
  async loadModel(manifest, options) {
    events.push('load');
    assert.equal(options.signal, manifest.signal, 'Construction must forward the original signal');
    await manifest.prepare?.();
  }
  async unload() { events.push('unload'); }
}

{
  const controller = new AbortController(); controller.abort();
  await assert.rejects(createInitializedPipeline(Pipeline, {}, { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(constructed, 0, 'Pre-aborted construction must not acquire a pipeline');
}

{
  const controller = new AbortController();
  const pipeline = await createInitializedPipeline(Pipeline, { signal: controller.signal }, { signal: controller.signal });
  assert.deepEqual(events, ['initialize', 'load']);
  await pipeline.unload();
}

{
  events.length = 0;
  const controller = new AbortController(), contexts = { signal: controller.signal };
  class ChangedContext extends Pipeline {
    async initialize() { contexts.signal = new AbortController().signal; controller.abort(); }
  }
  await assert.rejects(createInitializedPipeline(ChangedContext, {}, contexts), { name: 'AbortError' });
  assert.deepEqual(events, ['unload'], 'Later context mutation cannot replace the accepted cancellation signal');
}

{
  events.length = 0;
  const controller = new AbortController();
  let entered, resume;
  const ready = new Promise(resolve => { entered = resolve; });
  const pending = new Promise(resolve => { resume = resolve; });
  const reason = new Error('Opening cancelled');
  const opening = createInitializedPipeline(Pipeline, { signal: controller.signal,
    async prepare() { entered(); await pending; events.push('settled'); } },
  { signal: controller.signal }).catch(error => error);
  await ready;
  controller.abort(reason);
  await Promise.resolve();
  assert.deepEqual(events, ['initialize', 'load'], 'Do not close a program still preparing');
  resume();
  assert.equal(await opening, reason, 'Preserve the caller cancellation reason');
  assert.deepEqual(events, ['initialize', 'load', 'settled', 'unload']);
}

{
  let receipt = { expertPreparation: { status: 'deferred' } };
  const loader = { getLoadTiming: () => receipt };
  const state = { _preloadedWeights: { loader, layerWeights: new Map() }, weights: new Map(), stats: {},
    modelConfig: { useMoE: true }, runtimeConfig: { loading: { expertPreparation: { mode: 'eager-resident' } } } };
  await assert.rejects(_loadWeights.call(state), /Preloaded MoE weights must complete/);
  receipt = { expertPreparation: { status: 'complete' } };
  await _loadWeights.call(state);
  assert.equal(state.dopplerLoader, loader, 'Prepared weights retain their existing ownership');
}

console.log('pipeline-load-cancellation.test: signal forwarding, settled cleanup and preloaded admission passed (contract doubles)');
