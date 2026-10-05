import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Inject GPU ports only: exercise real partition orchestration and configured
// chunk boundaries. This is lifetime evidence, not numerical GPU evidence.
const target = new URL('../../src/inference/pipelines/text/partition-execution.js', import.meta.url).href;
let probe;
globalThis.partitionLifetime = {
  acquire(size) {
    if (probe.live.size >= 7) throw Error('allocation budget exceeded');
    const buffer = { size, bytes: new Uint8Array(size), __dopplerFakeGPUBuffer: true };
    probe.live.add(buffer); probe.peak = Math.max(probe.peak, probe.live.size); return buffer;
  },
  release(buffer) { assert(probe.live.delete(buffer), 'owned buffer released exactly once'); },
  recorder() {
    if (probe.fail === 'recorder' && probe.submissions > 0) throw Error('recorder failed');
    const owned = new Set(); let submitted = false;
    return {
      trackTemporaryBuffer(buffer) { assert(!submitted); owned.add(buffer); },
      getStats: () => ({ submitted }),
      getEncoder: () => ({ copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
        if (probe.fail === 'copy') throw Error('copy failed');
        destination.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + size), destinationOffset);
      } }),
      async submitAndWait() {
        submitted = true; probe.submissions++;
        for (const buffer of owned) globalThis.partitionLifetime.release(buffer);
        owned.clear();
        if (probe.fail === 'submit') throw Error('submit failed');
        if (probe.fail === 'cancel' || (probe.fail === 'later-cancel' && probe.submissions === 4)) probe.controller.abort();
      },
      resolveProfileTimings: async () => null,
      async abort() { for (const buffer of owned) globalThis.partitionLifetime.release(buffer); owned.clear(); },
    };
  },
  layer(layer, previous, tokens, prefill, context) {
    assert(probe.live.has(previous), 'carried activation remains owned');
    if (layer === 0 && probe.chunks) probe.chunks.push({ tokens, prefill, position: context.currentSeqLen });
    if (probe.fail === 'layer' && layer === 4) throw Error('layer failed');
    if (probe.fail === 'later-layer' && context.currentSeqLen === 3 && layer === 4) throw Error('later-layer failed');
    if (context.state.linearAttentionRuntime) {
      const prefix = context.state.linearAttentionRuntime.layers.get(layer);
      assert.equal(prefix?.seqLen ?? 0, context.currentSeqLen, 'each chunk extends the existing recurrent prefix');
      context.state.linearAttentionRuntime.layers.set(layer, { seqLen: context.currentSeqLen + tokens });
    }
    const output = globalThis.partitionLifetime.acquire(previous.size);
    output.bytes.set(previous.bytes);
    context.recorder.trackTemporaryBuffer(output);
    return output;
  },
};
const modules = {
  '../../../gpu/command-recorder.js': 'export const createCommandRecorder = () => partitionLifetime.recorder();',
  '../../../memory/buffer-pool.js': `export const acquireBuffer = size => partitionLifetime.acquire(size);
    export const releaseBuffer = buffer => partitionLifetime.release(buffer);
    export const uploadData = (buffer, source) => buffer.bytes.set(ArrayBuffer.isView(source)
      ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength) : new Uint8Array(source));
    export const readBuffer = async (buffer, size) => buffer.bytes.slice(0, size).buffer;`,
  './layer.js': 'export const processLayer = (...args) => partitionLifetime.layer(...args);',
  './generator/session-context.js': 'export const buildLayerContext = (state, recorder) => ({ state, recorder, currentSeqLen: state.currentSeqLen });',
  './layer-partition-contract.js': 'export const resolveLayerPartition = () => ({layerRange: [0, 11], hasEmbedding: false, hasLmHead: partitionLifetime.hasLmHead()});',
  './generator/logits-config.js': 'export const getLogitsWeights = () => ({}); export const getLogitsConfig = () => ({});',
  './logits/index.js': 'export const computeLogits = (...args) => partitionLifetime.logits(...args);',
};
globalThis.partitionLifetime.hasLmHead = () => probe.hasLmHead === true;
globalThis.partitionLifetime.logits = (_output, tokens) => {
  probe.logitsCalls.push(tokens);
  return { logitsBuffer: globalThis.partitionLifetime.acquire(16), vocabSize: 4 };
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
      inference: { session: { usePostFfnNextInputRMSNormPairFusion: false, prefillChunkLayers: 4, prefillTokenChunkSize: null } } } };
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
  state.runtimeConfig.inference.session.prefillTokenChunkSize = 3;
  state.modelConfig.layerTypes = Array(12).fill('linear_attention');
  const storage = Uint8Array.from({ length: 144 }, (_, index) => index);
  const input = { numTokens: 8, activationBytes: storage.subarray(8, 136) };
  for (const fail of [null, 'later-layer', 'later-cancel']) {
    for (const hasLmHead of [false, true]) {
      state.linearAttentionRuntime = { layers: new Map() };
      probe = { live: new Set(), peak: 0, submissions: 0, fail, controller: new AbortController(),
        chunks: [], logitsCalls: [], hasLmHead };
      const operation = executePartitionLayers(state, input, probe.controller.signal);
      if (fail) await assert.rejects(operation, fail === 'later-cancel' ? /abort/i : /later-layer failed/);
      else {
        const result = await operation;
        assert.deepEqual(probe.chunks, [
          { tokens: 3, prefill: true, position: 0 },
          { tokens: 3, prefill: true, position: 3 },
          { tokens: 2, prefill: true, position: 6 },
        ]);
        assert.equal(probe.submissions, 9, 'layer submissions remain bounded inside every token chunk');
        assert.deepEqual([...state.linearAttentionRuntime.layers.values()].map(value => value.seqLen), Array(12).fill(8));
        if (hasLmHead) {
          assert.deepEqual(probe.logitsCalls, [2], 'only the final token chunk computes logits');
          globalThis.partitionLifetime.release(result.logits.logitsBuffer);
        } else assert.deepEqual(new Uint8Array(result.activationBytes), input.activationBytes, 'byte assembly preserves row order and source view offsets');
      }
      assert.equal(state.currentSeqLen, 0, 'the resident operation owns the full-prompt sequence advance');
      assert.equal(probe.live.size, 0, `${fail || 'success'} token chunks settle every buffer`);
      assert(probe.peak <= 7);
    }
  }
  state.linearAttentionRuntime = { layers: new Map() };
  for (const badInput of [{ numTokens: 8, activationBytes: new ArrayBuffer(16) },
    { numTokens: 17, activationBytes: new ArrayBuffer(272) }]) {
    probe = { live: new Set(), peak: 0, submissions: 0, controller: new AbortController() };
    await assert.rejects(executePartitionLayers(state, badInput, probe.controller.signal), /byte length|sequence allocation/);
    assert.equal(probe.submissions, 0);
    assert.equal(probe.live.size, 0);
  }
} finally { hooks.deregister(); delete globalThis.partitionLifetime; }
console.log('partition-prefill-lifetime: bounded chunks, carry, failures and cancellation passed');
