import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { setDevice } from '../../src/gpu/device.js';

installNodeFileFetchShim();
const { _runDecodeLoop } = await import('../../src/inference/pipelines/text/generator/decode-runtime.js');

// Keep the real delivery loop. Synthetic computation lets cancellation occur at
// exact suspension boundaries, without conflating transport or GPU completion.
setDevice({ features: new Set(), limits: { maxStorageBufferBindingSize: 1048576,
  maxBufferSize: 1048576, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16384 },
queue: { submit() {} }, createBuffer() { return { destroy() {} }; }, createBindGroup() { return {}; } }, { platformConfig: null });
try {
  for (const mode of ['batch', 'single', 'speculation', 'failed-batch']) {
    for (const boundary of ['pending', 'delivered', 'callback', 'speculative-pending']) {
      if (boundary === 'speculative-pending' && mode !== 'speculation') continue;
      if (mode === 'failed-batch' && boundary !== 'pending') continue;
      for (const emitMode of ['token', 'text']) {
        const abort = new AbortController();
        const ids = [1], callbacks = [], batches = [];
        let computations = 0;
        const state = { useGPU: true, modelConfig: { hiddenSize: 4, numLayers: 1, layerTypes: ['full_attention'] },
          runtimeConfig: createDopplerConfig().runtime, weights: new Map(), kvCache: { layout: 'contiguous' },
          currentSeqLen: 1, batchingStats: {}, stats: {},
          tokenizer: { getSpecialTokens: () => ({ eos: 9, pad: 0 }) } };
        const compute = async () => {
          computations++;
          if (boundary === 'pending' || (boundary === 'speculative-pending' && computations === 2)) abort.abort();
          return 2;
        };
        const pipeline = { _state: state, _hasFinitenessFallbackWindow: () => false,
          _consumeFinitenessFallbackToken() {}, _shouldUseFinitenessFallback: () => false,
          _recordStopReason(reason) { state.stats.stopReason = reason; },
          async _generateNTokensGPU() {
            await compute();
            if (mode === 'failed-batch') throw new Error('Cancelled submitted batch');
            return { tokens: [2, 3, 4, 5], actualCount: 4 };
          },
          _decodeNextTokenViaLogits: compute };
        const opts = { maxTokens: 8, stopSequences: [], suppressTokenIds: [],
          speculation: mode === 'speculation' ? { mode: 'self', tokens: 3 } : null,
          executionPlan: { batchSize: mode.includes('batch') ? 4 : 1, readbackInterval: 1,
            disableMultiTokenDecode: false, disableCommandBatching: false,
            maxBatchDecodeTokens: null, activationDtype: 'f32' } };
        const stream = _runDecodeLoop.call(pipeline, ids, opts, { signal: abort.signal,
          onToken(token) { callbacks.push(token); if (boundary === 'callback') abort.abort(); },
          onBatch(batch) { batches.push(batch.map(token => token.id)); } },
        { stopTokenIds: [9], eosToken: 9, stopSequenceStart: 0, decodeToken: String, emitMode });
        const first = await stream.next();
        if (boundary === 'pending') assert.equal(first.done, true);
        else {
          assert.deepEqual(first, { done: false, value: emitMode === 'token' ? 2 : '2' });
          if (boundary === 'delivered') abort.abort();
          assert.equal((await stream.next()).done, true, `${mode}/${boundary}/${emitMode}`);
        }
        assert.deepEqual(ids, boundary === 'pending' ? [1] : [1, 2]);
        assert.deepEqual(callbacks, boundary === 'pending' ? [] : [2]);
        assert.deepEqual(batches.flat(), mode === 'batch' && boundary !== 'pending' ? [2] : []);
        assert.equal(computations, boundary === 'speculative-pending' ? 2 : 1, 'Abort must not start a fallback or speculative dispatch');
        assert.equal(state.stats.stopReason, 'aborted');
      }
    }
  }
} finally { setDevice(null); }
console.log('generator-abort-delivery.test: ok');
