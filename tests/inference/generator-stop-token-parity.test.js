import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { setDevice } from '../../src/gpu/device.js';

installNodeFileFetchShim();
const { _runDecodeLoop } = await import('../../src/inference/pipelines/text/generator/decode-runtime.js');

// Synthetic token production exercises the actual decode-loop ownership boundary.
// The terminal token belongs to the transcript; tokens after it never do.
setDevice({ features: new Set(), limits: { maxStorageBufferBindingSize: 1048576,
  maxBufferSize: 1048576, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16384 },
queue: { submit() {} }, createBuffer() { return { destroy() {} }; }, createBindGroup() { return {}; } }, { platformConfig: null });
try {
  for (const stop of [9, 8]) {
    for (const batchSize of [1, 4]) {
      const runtimeConfig = createDopplerConfig().runtime;
      const ids = [1];
      const callbacks = [];
      const state = { useGPU: true, modelConfig: { hiddenSize: 4, numLayers: 1, layerTypes: ['full_attention'] },
        runtimeConfig, weights: new Map(), kvCache: { layout: 'contiguous' }, currentSeqLen: 1,
        batchingStats: {}, stats: {}, tokenizer: { getSpecialTokens: () => ({ eos: 9, pad: 0 }) } };
      let step = 0;
      const pipeline = { _state: state, _hasFinitenessFallbackWindow: () => false,
        _consumeFinitenessFallbackToken() {}, _shouldUseFinitenessFallback: () => false,
        _recordStopReason(reason, tokenId) { state.stats.stopReason = reason; state.stats.stopTokenId = tokenId; },
        async _generateNTokensGPU() { return { tokens: [2, stop, 7], actualCount: 3 }; },
        async _decodeNextTokenViaLogits() { return [2, stop][step++]; } };
      const opts = { maxTokens: 6, stopSequences: [], suppressTokenIds: [], speculation: null,
        executionPlan: { batchSize, readbackInterval: 1, disableMultiTokenDecode: false,
          disableCommandBatching: false, maxBatchDecodeTokens: null, activationDtype: 'f32' } };
      const emitted = [];
      for await (const token of _runDecodeLoop.call(pipeline, ids, opts,
        { onToken: token => callbacks.push(token) }, { stopTokenIds: [8], eosToken: 9,
          stopSequenceStart: 0, decodeToken: String, emitMode: 'token' })) emitted.push(token);
      assert.deepEqual(emitted, [2, stop], `batchSize=${batchSize}, stop=${stop}`);
      assert.deepEqual(callbacks, emitted);
      assert.deepEqual(ids, [1, 2, stop]);
      assert.equal(state.stats.stopReason, 'stop-token');
      assert.equal(state.stats.stopTokenId, stop);
      assert.equal(state.stats.tokensGenerated, 3);
    }
  }
} finally { setDevice(null); }
console.log('generator-stop-token-parity.test: ok');
