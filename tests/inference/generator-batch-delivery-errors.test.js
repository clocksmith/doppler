import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { setDevice } from '../../src/gpu/device.js';

installNodeFileFetchShim();
const { _runDecodeLoop } = await import('../../src/inference/pipelines/text/generator/decode-runtime.js');
setDevice({ features: new Set(), limits: { maxStorageBufferBindingSize: 1048576,
  maxBufferSize: 1048576, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16384 },
queue: { submit() {} }, createBuffer() { return { destroy() {} }; }, createBindGroup() { return {}; } }, { platformConfig: null });
try {
  for (const boundary of ['onToken', 'onBatch', 'decodeToken', 'consumerThrow', 'gpuFailure']) {
    const sentinel = new Error(`sentinel:${boundary}`);
    const ids = [1];
    let fallbackCalls = 0;
    const state = { useGPU: true, modelConfig: { hiddenSize: 4, numLayers: 1, layerTypes: ['full_attention'] },
      runtimeConfig: createDopplerConfig().runtime, weights: new Map(), kvCache: { layout: 'contiguous' },
      currentSeqLen: 1, batchingStats: {}, stats: {}, tokenizer: { getSpecialTokens: () => ({ eos: 9, pad: 0 }) } };
    const pipeline = { _state: state, _hasFinitenessFallbackWindow: () => false,
      _consumeFinitenessFallbackToken() {}, _shouldUseFinitenessFallback: () => false,
      _recordStopReason(reason) { state.stats.stopReason = reason; },
      async _generateNTokensGPU() {
        if (boundary === 'gpuFailure') throw sentinel;
        return { tokens: [2, 3, 4, 5], actualCount: 4 };
      },
      async _decodeNextTokenViaLogits() { fallbackCalls++; return 9; } };
    const opts = { maxTokens: 8, stopSequences: [], suppressTokenIds: [], speculation: null,
      executionPlan: { batchSize: 4, readbackInterval: 1, disableMultiTokenDecode: false,
        disableCommandBatching: false, maxBatchDecodeTokens: null, activationDtype: 'f32' } };
    const fail = () => { throw sentinel; };
    const stream = _runDecodeLoop.call(pipeline, ids, opts,
      boundary === 'onToken' || boundary === 'onBatch' ? { [boundary]: fail } : {},
      { stopTokenIds: [9], eosToken: 9, stopSequenceStart: 0,
        decodeToken: boundary === 'decodeToken' ? fail : String,
        emitMode: boundary === 'decodeToken' ? 'text' : 'token' });
    if (boundary === 'gpuFailure') {
      assert.deepEqual(await Array.fromAsync(stream), [9]);
      assert.equal(fallbackCalls, 1, 'GPU computation failure retains its existing fallback');
      assert.equal(state.stats.stopReason, 'stop-token');
    } else {
      if (boundary === 'consumerThrow') {
        assert.equal((await stream.next()).value, 2);
        await assert.rejects(stream.throw(sentinel), error => error === sentinel);
      } else await assert.rejects(Array.fromAsync(stream), error => error === sentinel);
      assert.equal(fallbackCalls, 0, `${boundary} must not trigger another GPU decode`);
    }
  }
} finally { setDevice(null); }
console.log('generator-batch-delivery-errors.test: ok');
