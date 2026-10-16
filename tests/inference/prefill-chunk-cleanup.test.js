import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Real prefill and CommandRecorder lifecycle; only GPU operations and tensor
// kernels are substituted. No numerical or hardware qualification is implied.
const target = new URL('../../src/inference/pipelines/text/generator/prefill-runtime.js', import.meta.url).href;
const recorderUrl = new URL('../../src/gpu/command-recorder.js', import.meta.url).href;
const active = new Set();
const probe = { active, gates: [], submissions: 0, gateCompletion: false };
globalThis.prefillChunkProbe = probe;
probe.acquire = () => {
  const buffer = { __dopplerFakeGPUBuffer: true, size: 32, usage: 0 };
  active.add(buffer);
  return buffer;
};
probe.device = {
  createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish: () => ({}) }),
  queue: {
    submit() { probe.submissions += 1; },
    onSubmittedWorkDone() {
      if (!probe.gateCompletion) return Promise.resolve();
      const gate = Promise.withResolvers();
      probe.gates.push(gate);
      return gate.promise;
    },
  },
};
const poolModule = `
  const p = globalThis.prefillChunkProbe;
  export const acquireBuffer = () => p.acquire();
  export const releaseBuffer = buffer => p.active.delete(buffer);
  export const discardBuffer = releaseBuffer;
  export const isBufferActive = buffer => p.active.has(buffer);
  export const isPersistentBuffer = () => false;
  export const getBufferPool = () => ({ settleDeferredDestruction: async () => {} });
  export const readBuffer = () => { throw Error('unexpected readback'); };
  export const readBufferSlice = readBuffer; export const uploadData = readBuffer;
`;
const modules = {
  '../../../../gpu/device.js': 'export const getDevice = () => globalThis.prefillChunkProbe.device; export const setTrackSubmits = () => {};',
  '../embed.js': 'export const embed = async () => ({ buffer: globalThis.prefillChunkProbe.acquire(), dtype: "f32", shape: [2, 4] });',
  '../layer.js': `export const processLayer = async (layer, input, tokens, prefill, context) => {
    const p = globalThis.prefillChunkProbe;
    if (layer === 2 && p.fail) throw Error('layer failed');
    if (layer === 1 && p.controller) p.controller.abort();
    const temporary = p.acquire();
    context.recorder.trackTemporaryBuffer(temporary);
    return p.acquire();
  };`,
  '../logits/index.js': 'export const computeLogits = async () => ({ logitsBuffer: "caller-owned" }); export const recordLogitsGPU = computeLogits; export const extractLastPositionLogits = value => value;',
  './session-context.js': 'export const buildLayerContext = () => ({ useGPU: true }); export const debugCheckBuffer = () => {};',
  './logits-config.js': 'export const getLogitsWeights = () => ({}); export const getLogitsConfig = () => ({});',
  '../per-layer-inputs.js': 'export const preparePerLayerInputs = async () => null;',
  './prefill-observation.js': 'export const probePrefillEmbedding = async () => {}; export const tracePrefillEmbeddingIds = () => {};',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  let source;
  if ([target, recorderUrl].includes(context.parentURL) && specifier.endsWith('/buffer-pool.js')) source = poolModule;
  else if (context.parentURL === target) source = modules[specifier];
  if (source) return { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { _prefill } = await import(target);
  const host = { _state: {
    modelConfig: { hiddenSize: 4, vocabSize: 4, numLayers: 3 },
    weights: new Map([['embed', new Float32Array(16)]]), convLayerStates: new Map(), stats: {}, currentSeqLen: 0,
    useGPU: true, runtimeConfig: { inference: { session: { prefillChunkLayers: 1, prefillChunkSubmitMode: 'async' } }, shared: { debug: { probes: null } } },
  }, _getEffectiveActivationDtype: () => 'f32' };
  for (let repeat = 0; repeat < 3; repeat++) {
    const result = await _prefill.call(host, [1, 2], { _returnGpuLogits: true });
    assert.equal(result.logitsBuffer, 'caller-owned');
    assert.equal(active.size, 0, 'Every chunk releases its intermediates, across repeated requests');
  }

  // Cancellation and recording failure must keep submitted buffers owned until
  // completion. A failed prefix must not let later submitted work escape cleanup.
  for (const failure of ['layer', 'cancel', 'completion', 'transfer-completion']) {
    probe.fail = failure === 'layer';
    probe.controller = failure === 'cancel' ? new AbortController() : null;
    probe.gateCompletion = true;
    probe.gates = [];
    let settled = false;
    const result = _prefill.call(host, [1, 2], {
      _returnGpuLogits: true, _returnHidden: failure === 'transfer-completion', signal: probe.controller?.signal,
    }).then(value => ({ value }), error => ({ error })).finally(() => { settled = true; });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(probe.gates.length >= 2, true, 'Two prefixes were submitted without waiting');
    assert.equal(settled, false);
    assert(active.size > 0, 'Submitted buffers stay owned until GPU completion');
    if (failure.includes('completion')) probe.gates[0].reject(Error('GPU completion failed'));
    else probe.gates[0].resolve();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(settled, false, 'All submitted work must settle, even after a prefix fails');
    for (const gate of probe.gates.slice(1)) gate.resolve();
    const outcome = await result;
    assert.match(String(outcome.error), failure === 'layer' ? /layer failed/ : failure === 'cancel' ? /abort/i : /GPU completion failed/);
    assert.equal(active.size, 0, `${failure} releases all chunk resources`);
  }
  probe.gateCompletion = false;
  probe.fail = false;
  probe.controller = null;
  const hidden = await _prefill.call(host, [1, 2], { _returnHidden: true });
  assert(active.has(hidden.currentHiddenBuffer), 'Successful transfer preserves caller ownership');
  await hidden.currentRecorder.submitAndWait();
  active.delete(hidden.currentHiddenBuffer);
  assert.equal(active.size, 0, 'Earlier chunks cannot leak through hidden-state transfer');
} finally {
  hooks.deregister();
  delete globalThis.prefillChunkProbe;
}
console.log('prefill-chunk-cleanup: passed');
