import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Import-only substitutions: exercise the unchanged prefill orchestrator with
// deterministic failures at the layer and logits boundaries, without a GPU.
const target = new URL('../../src/inference/pipelines/text/generator/prefill-runtime.js', import.meta.url).href;
const hidden = { __dopplerFakeGPUBuffer: true, size: 32, usage: 0 };
globalThis.prefillOwnershipProbe = { hidden, released: [], fail: 'logits' };
const modules = {
  '../../../../gpu/device.js': 'export const getDevice = () => null; export const setTrackSubmits = () => {};',
  '../../../../memory/buffer-pool.js': `export const releaseBuffer = buffer => globalThis.prefillOwnershipProbe.released.push(buffer);
    export const acquireBuffer = () => { throw Error('unexpected acquire'); }; export const readBuffer = acquireBuffer;
    export const readBufferSlice = acquireBuffer; export const uploadData = acquireBuffer;`,
  '../embed.js': 'export const embed = async () => ({ buffer: globalThis.prefillOwnershipProbe.hidden, dtype: "f32", shape: [2, 4] });',
  '../layer.js': 'export const processLayer = async () => { if (globalThis.prefillOwnershipProbe.fail === "layer") throw Error("layer failed"); return globalThis.prefillOwnershipProbe.hidden; };',
  '../logits/index.js': `export const computeLogits = async () => { if(globalThis.prefillOwnershipProbe.fail === 'logits') throw Error('logits failed'); return { logitsBuffer: 'caller-owned' }; };
    export const recordLogitsGPU = computeLogits; export const extractLastPositionLogits = value => value;`,
  './session-context.js': 'export const buildLayerContext = () => ({ useGPU: true }); export const debugCheckBuffer = () => {};',
  './logits-config.js': 'export const getLogitsWeights = () => ({}); export const getLogitsConfig = () => ({});',
  '../per-layer-inputs.js': 'export const preparePerLayerInputs = async () => null;',
  './prefill-observation.js': 'export const probePrefillEmbedding = async () => {}; export const tracePrefillEmbeddingIds = () => {};',
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
} finally { hooks.deregister(); delete globalThis.prefillOwnershipProbe; }
console.log('prefill-output-ownership: passed');
