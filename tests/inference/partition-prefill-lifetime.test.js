import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Inject GPU ports only: exercise real partition orchestration and configured
// chunk boundaries. This is lifetime evidence, not numerical GPU evidence.
const target = new URL('../../src/inference/pipelines/text/partition-execution.js', import.meta.url).href;
let probe;
globalThis.partitionLifetime = {
  acquire(size) {
    if (probe.live.size >= 7) throw Error('allocation budget exceeded');
    const buffer = { size, __dopplerFakeGPUBuffer: true };
    probe.live.add(buffer); probe.peak = Math.max(probe.peak, probe.live.size); return buffer;
  },
  release(buffer) { assert(probe.live.delete(buffer), 'owned buffer released exactly once'); },
  recorder() {
    if (probe.fail === 'recorder' && probe.submissions > 0) throw Error('recorder failed');
    const owned = new Set(); let submitted = false;
    return {
      trackTemporaryBuffer(buffer) { assert(!submitted); owned.add(buffer); },
      getStats: () => ({ submitted }),
      getEncoder: () => ({ copyBufferToBuffer() { if (probe.fail === 'copy') throw Error('copy failed'); } }),
      async submitAndWait() {
        submitted = true; probe.submissions++;
        for (const buffer of owned) globalThis.partitionLifetime.release(buffer);
        owned.clear();
        if (probe.fail === 'submit') throw Error('submit failed');
        if (probe.fail === 'cancel') probe.controller.abort();
      },
      resolveProfileTimings: async () => null,
      async abort() { for (const buffer of owned) globalThis.partitionLifetime.release(buffer); owned.clear(); },
    };
  },
  layer(layer, previous, _tokens, _prefill, context) {
    assert(probe.live.has(previous), 'carried activation remains owned');
    if (probe.fail === 'layer' && layer === 4) throw Error('layer failed');
    const output = globalThis.partitionLifetime.acquire(previous.size);
    context.recorder.trackTemporaryBuffer(output);
    return output;
  },
};
const modules = {
  '../../../gpu/command-recorder.js': 'export const createCommandRecorder = () => partitionLifetime.recorder();',
  '../../../memory/buffer-pool.js': `export const acquireBuffer = size => partitionLifetime.acquire(size);
    export const releaseBuffer = buffer => partitionLifetime.release(buffer);
    export const uploadData = () => {}; export const readBuffer = async (buffer, size) => new ArrayBuffer(size);`,
  './layer.js': 'export const processLayer = (...args) => partitionLifetime.layer(...args);',
  './generator/session-context.js': 'export const buildLayerContext = (_state, recorder) => ({ recorder });',
  './layer-partition-contract.js': 'export const resolveLayerPartition = () => ({layerRange: [0, 11], hasEmbedding: false, hasLmHead: false});',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { executePartitionLayers } = await import(target);
  const state = { useGPU: true, manifest: {}, modelPartition: { plan: { activationDtype: 'f32' } },
    modelConfig: { hiddenSize: 4, useMoE: false, numKvSharedLayers: 0, hiddenSizePerLayerInput: null,
      decodeStrategy: 'incremental', causalAttention: true }, currentSeqLen: 0, kvCache: { maxSeqLen: 16 },
    executionPlanState: { primaryPlan: { activationDtype: 'f32', finitenessGuardEnabled: false } },
    runtimeConfig: { shared: { debug: { profiler: { enabled: false } } },
      inference: { session: { usePostFfnNextInputRMSNormPairFusion: false, prefillChunkLayers: 4 } } } };
  for (const fail of [null, 'layer', 'submit', 'copy', 'recorder', 'cancel']) {
    probe = { live: new Set(), peak: 0, submissions: 0, fail, controller: new AbortController() };
    const operation = executePartitionLayers(state, { numTokens: 2, activationBytes: new ArrayBuffer(32) }, probe.controller.signal);
    if (fail) await assert.rejects(operation, fail === 'cancel' ? /abort/i : new RegExp(`${fail} failed`));
    else {
      assert.equal((await operation).activationBytes.byteLength, 32);
      assert.equal(probe.submissions, 3);
      assert(probe.peak <= 7);
    }
    assert.equal(probe.live.size, 0, `${fail || 'success'} settles all owned buffers`);
  }
} finally { hooks.deregister(); delete globalThis.partitionLifetime; }
console.log('partition-prefill-lifetime: bounded chunks, carry, failures and cancellation passed');
