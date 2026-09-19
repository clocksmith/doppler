import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
const target = new URL('../../src/inference/pipelines/text/generator/decode.js', import.meta.url).href;
const hidden = { __dopplerFakeGPUBuffer: true, size: 16, usage: 0 };
const logits = { __dopplerFakeGPUBuffer: true, size: 16, usage: 0 };
globalThis.decodeOwnershipProbe = { hidden, logits, released: [], fail: 'readback' };
const modules = {
  '../../../../gpu/device.js': 'export const getDevice = () => null; export const setTrackSubmits = () => {};',
  '../../../../memory/buffer-pool.js': `export const releaseBuffer = buffer => globalThis.decodeOwnershipProbe.released.push(buffer);
    export const readBuffer = async () => { throw Error('readback failed'); };`,
  '../embed.js': 'export const embed = async () => ({ buffer: globalThis.decodeOwnershipProbe.hidden, dtype: "f32", shape: [1, 4] });',
  '../layer.js': 'export const hasConvLayers = () => false; export const processLayer = async () => { if (globalThis.decodeOwnershipProbe.fail === "layer") throw Error("layer failed"); return globalThis.decodeOwnershipProbe.hidden; };',
  '../logits/index.js': `export const computeLogitsGPU = async () => ({ logitsBuffer: globalThis.decodeOwnershipProbe.logits, logitsDtype: 'f32', vocabSize: 4 });
    export const computeLogits = computeLogitsGPU; export const recordLogitsGPU = computeLogitsGPU;
    export const extractLastPositionLogits = value => value;`,
  '../per-layer-inputs.js': 'export const preparePerLayerInputs = async () => null; export const prefetchPerLayerRow = () => {};',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { decodeStepLogits } = await import(target);
  const state = { modelConfig: { hiddenSize: 4, vocabSize: 4, numLayers: 1 },
    weights: new Map([['embed', new Float32Array(16)]]), stats: {}, currentSeqLen: 0,
    decodeBuffers: { resetPingPong() {}, swapPingPong() {}, getHiddenBuffer: () => null, getOutputHiddenBuffer: () => null },
    useGPU: true, runtimeConfig: { inference: { compute: { activationDtype: 'f32' } }, shared: { debug: { probes: null } } } };
  const helpers = { buildLayerContext: () => ({}), getLogitsWeights: () => ({}), getLogitsConfig: () => ({}) };
  for (const failure of ['layer', 'readback']) {
    globalThis.decodeOwnershipProbe.fail = failure;
    globalThis.decodeOwnershipProbe.released.length = 0;
    await assert.rejects(decodeStepLogits(state, [1], {}, helpers), new RegExp(`${failure} failed`));
    assert.deepEqual(globalThis.decodeOwnershipProbe.released, failure === 'layer' ? [hidden] : [hidden, logits]);
  }
  globalThis.decodeOwnershipProbe.fail = null;
  globalThis.decodeOwnershipProbe.released.length = 0;
  const result = await decodeStepLogits(state, [1], { _returnGpuLogits: true }, helpers);
  assert.equal(result.logitsBuffer, logits);
  assert.deepEqual(globalThis.decodeOwnershipProbe.released, [hidden]);
  state.decodeBuffers.getHiddenBuffer = () => hidden;
  globalThis.decodeOwnershipProbe.released.length = 0;
  await decodeStepLogits(state, [1], { _returnGpuLogits: true }, helpers);
  assert.deepEqual(globalThis.decodeOwnershipProbe.released, [], 'session-owned ping-pong and returned logits stay live');
} finally { hooks.deregister(); delete globalThis.decodeOwnershipProbe; }
console.log('decode-output-ownership: passed');
