import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Import-only substitutions: exercise the unchanged prefill orchestrator with
// deterministic failures at the layer and logits boundaries, without a GPU.
const target = new URL('../../src/inference/pipelines/text/generator/prefill-runtime.js', import.meta.url).href;
const hidden = { __dopplerFakeGPUBuffer: true, size: 32, usage: 0 };
globalThis.prefillOwnershipProbe = { hidden, released: [], fail: 'logits' };
const modules = {
  '../../../../gpu/device.js': 'export const getDevice = () => globalThis.prefillOwnershipProbe.device ?? null; export const setTrackSubmits = () => {};',
  '../../../../memory/buffer-pool.js': `export const releaseBuffer = buffer => globalThis.prefillOwnershipProbe.released.push(buffer);
    export const getBufferPool = device => { const probe = globalThis.prefillOwnershipProbe; probe.poolDevice = device; return probe.pool; };
    export const acquireBuffer = () => { throw Error('unexpected acquire'); }; export const readBuffer = acquireBuffer;
    export const readBufferSlice = acquireBuffer; export const uploadData = acquireBuffer;`,
  '../../../../gpu/command-recorder.js': `export const createCommandRecorder = () => globalThis.prefillOwnershipProbe.recorder;
    export const createProfilingRecorder = createCommandRecorder; export class CommandRecorder {}`,
  '../embed.js': 'export const embed = async () => ({ buffer: globalThis.prefillOwnershipProbe.hidden, dtype: "f32", shape: [2, 4] });',
  '../layer.js': 'export const processLayer = async () => { if (globalThis.prefillOwnershipProbe.fail === "layer") throw Error("layer failed"); return globalThis.prefillOwnershipProbe.hidden; };',
  '../logits/index.js': `export const computeLogits = async () => { if(globalThis.prefillOwnershipProbe.fail === 'logits') throw Error('logits failed'); return { logitsBuffer: 'caller-owned' }; };
    export const recordLogitsGPU = computeLogits; export const extractLastPositionLogits = value => value;`,
  './session-context.js': 'export const buildLayerContext = () => ({ useGPU: true }); export const debugCheckBuffer = () => {};',
  './logits-config.js': 'export const getLogitsWeights = () => ({}); export const getLogitsConfig = () => ({});',
  '../per-layer-inputs.js': 'export const preparePerLayerInputs = async () => null;',
  './prefill-observation.js': 'export const probePrefillEmbedding = async () => {}; export const tracePrefillEmbeddingIds = () => {}; export const logPrefillCacheState = () => {};',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { _prefill } = await import(target);
  const host = { _state: { modelConfig: { hiddenSize: 4, vocabSize: 4, numLayers: 1 },
    weights: new Map([['embed', new Float32Array(16)]]), convLayerStates: new Map(), stats: {}, currentSeqLen: 0,
    useGPU: true, runtimeConfig: { inference: { session: { prefillChunkLayers: 1 } }, shared: { debug: { probes: null } } } },
    _getEffectiveActivationDtype: () => 'f32' };
  for (const failure of ['layer', 'logits']) {
    globalThis.prefillOwnershipProbe.fail = failure;
    globalThis.prefillOwnershipProbe.released.length = 0;
    await assert.rejects(_prefill.call(host, [1, 2], { _returnGpuLogits: true }), new RegExp(`${failure} failed`));
    assert.deepEqual(globalThis.prefillOwnershipProbe.released, [hidden], `${failure} failure releases the owned hidden buffer once`);
  }
  globalThis.prefillOwnershipProbe.fail = null;
  globalThis.prefillOwnershipProbe.released.length = 0;
  const result = await _prefill.call(host, [1, 2], { _returnGpuLogits: true });
  assert.equal(result.logitsBuffer, 'caller-owned');
  assert.deepEqual(globalThis.prefillOwnershipProbe.released, [hidden]);
  globalThis.prefillOwnershipProbe.released.length = 0;
  const intermediate = await _prefill.call(host, [1, 2], { _returnHidden: true });
  assert.equal(intermediate.currentHiddenBuffer, hidden);
  assert.deepEqual(globalThis.prefillOwnershipProbe.released, [], 'chunk caller owns transferred hidden state');

  // The public failure must await recorder ownership release before retiring the
  // captured pool. Changing the global device during cleanup must not redirect it.
  const probe = globalThis.prefillOwnershipProbe;
  const originalDevice = {};
  const abortGate = Promise.withResolvers();
  const retirementGate = Promise.withResolvers();
  const abortStarted = Promise.withResolvers();
  const retirementStarted = Promise.withResolvers();
  const order = [];
  probe.device = originalDevice;
  probe.fail = 'layer';
  probe.pool = { async settleDeferredDestruction() {
    order.push('retire'); retirementStarted.resolve(); await retirementGate.promise;
  } };
  probe.recorder = { async abort() {
    order.push('abort'); abortStarted.resolve(); await abortGate.promise; order.push('abort-done');
  } };
  let completed = false;
  const failed = _prefill.call(host, [1, 2], { _returnGpuLogits: true })
    .then(() => { throw Error('Expected layer failure'); }, error => error)
    .finally(() => { completed = true; });
  await abortStarted.promise;
  await Promise.resolve();
  assert.equal(completed, false, 'Failed prefill must await its recorder abort');
  probe.device = {};
  probe.pool = { settleDeferredDestruction() { throw Error('Wrong device pool'); } };
  abortGate.resolve();
  const retirementReached = await Promise.race([
    retirementStarted.promise.then(() => true), failed.then(() => false),
  ]);
  assert.equal(retirementReached, true, 'Failure must retire buffers before becoming observable');
  assert.equal(probe.poolDevice, originalDevice, 'Pool ownership is captured before async work');
  assert.equal(completed, false, 'Failed prefill must await deferred retirement');
  retirementGate.resolve();
  assert.match((await failed).message, /layer failed/);
  assert.deepEqual(order, ['abort', 'abort-done', 'retire']);

  const abortError = Error('recorder cleanup failed');
  const retirementError = Error('retirement failed');
  probe.device = originalDevice;
  probe.recorder = { async abort() { throw abortError; } };
  probe.pool = { async settleDeferredDestruction() { throw retirementError; } };
  await assert.rejects(_prefill.call(host, [1, 2], { _returnGpuLogits: true }), error => {
    assert(error instanceof AggregateError);
    assert.match(error.errors[0].message, /layer failed/);
    assert.equal(error.cause, error.errors[0]);
    assert.equal(error.errors[1], abortError);
    assert.equal(error.errors[2], retirementError);
    return true;
  });
  probe.device = null;
  probe.fail = null;
  assert.equal((await _prefill.call(host, [1, 2], { _returnGpuLogits: true })).logitsBuffer, 'caller-owned');
} finally { hooks.deregister(); delete globalThis.prefillOwnershipProbe; }
console.log('prefill-output-ownership: passed');
