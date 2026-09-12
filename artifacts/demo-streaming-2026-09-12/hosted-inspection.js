import assert from 'node:assert/strict';

import { createModelHandle } from 'file:///home/x/deco/d4da/sites/d4da-com/doppler/src/client/runtime/model-session.js';

function canonicalize(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((entry) => canonicalize(entry)).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalize(value[key])}`
  )).join(',')}}`;
}

async function hashValue(value) {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(canonicalize(value))
  );
  const hex = Array.from(new Uint8Array(digest), (byte) => (
    byte.toString(16).padStart(2, '0')
  )).join('');
  return `sha256:${hex}`;
}

const runtimeConfig = {
  inference: {
    generation: {
      maxTokens: 32,
      useSpeculative: false,
    },
    sampling: {
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      repetitionPenalty: 1.1,
      repetitionPenaltyWindow: 64,
      presencePenalty: 0,
      greedyThreshold: 0,
      suppressSpecialTokens: true,
      suppressSpecialLikeTokens: true,
      suppressTokenIds: [],
    },
    chatTemplate: {
      enabled: true,
    },
  },
};

const stats = {
  kernelPathId: 'qwen35-webgpu',
  kernelPathSource: 'execution-v1',
  executionPlan: {
    primary: {
      id: 'primary',
      kernelPathId: 'qwen35-webgpu',
      kernelPathSource: 'execution-v1',
      activationDtype: 'f16',
    },
    fallback: null,
    activePlanIdAtStart: 'primary',
    finalActivePlanId: 'primary',
    transitions: [],
  },
};

const pipeline = {
  modelConfig: {
    chatTemplateEnabled: true,
  },
  manifest: {
    inference: {
      chatTemplate: {
        enabled: true,
      },
    },
  },
  runtimeConfig,
  resolvedRuntimeSession: {
    id: `sha256:${'b'.repeat(64)}`,
  },
  isLoaded: true,
  tokenizer: {
    encode() {
      return [11, 12, 13];
    },
    decode(tokenIds) {
      return tokenIds.map((tokenId) => `[${tokenId}]`).join('');
    },
  },
  async generateTokenIds(prompt, options) {
    if (prompt === 'hello') {
      assert.deepEqual(options, {
        maxTokens: 4,
        temperature: 0,
        topK: 1,
        topP: 1,
        useChatTemplate: false,
      });
    } else {
      assert.deepEqual(prompt, [{ role: 'user', content: 'hello' }]);
      assert.deepEqual(options, { maxTokens: 2 });
    }
    return {
      tokenIds: [101, 202],
      stats,
    };
  },
  getKernelCapabilities() {
    return {
      adapterInfo: {
        vendor: 'test-vendor',
        architecture: 'test-architecture',
        device: 'test-device',
        description: 'test adapter',
      },
      hasF16: true,
      hasSubgroups: false,
      maxBufferSize: 4096,
      deviceEpoch: 3,
    };
  },
  generate() {},
  async embed(prompt) {
    assert.equal(prompt, 'embedding input');
    return {
      embedding: new Float32Array([0.25, 0.75]),
      tokens: [31, 32],
      seqLen: 2,
      embeddingMode: 'mean',
      phase: null,
    };
  },
  embedBatch() {},
  embedImage() {},
  embedAudio() {},
  transcribeImage() {},
  transcribeAudio() {},
  transcribeVideo() {},
  unload() {},
};

// Inspection events arrive before completion without changing sampling/batching.
let releaseStream;
let emitToken;
let observedOptions;
let streamCalls = 0;
const streamHandle = createModelHandle({
  ...pipeline,
  manifest: { ...pipeline.manifest, tokenizer: { type: 'test', digest: 'fixture' } },
  async generateTokenIds(_prompt, options) {
    streamCalls++;
    observedOptions = options;
    emitToken = options.onToken;
    options.onToken?.(101, '');
    await new Promise((resolve) => { releaseStream = resolve; });
    options.onToken?.(202, '');
    return { tokenIds: [101, 202], stats };
  },
}, { modelId: 'stream-test', manifestHash: 'a'.repeat(64) });

await assert.rejects(streamHandle.inspect.generate('test', { onEvent: 'bad' }), /onEvent must be a function/);
await assert.rejects(streamHandle.inspect.generate('test', { generation: { onToken() {} } }), /inspection owns/);
assert.equal(streamCalls, 0);
const events = [];
const streamedResult = streamHandle.inspect.generate('test', { onEvent: (event) => events.push(event) });
assert.deepEqual(events, [{ type: 'token', tokenId: 101, index: 0 }]);
assert.equal(observedOptions.disableCommandBatching, undefined);
assert.equal(observedOptions.profile, undefined);
assert.equal(observedOptions.onLogits, undefined);
releaseStream();
const streamedReceipt = await streamedResult;
assert.deepEqual(events.slice(0, 2), [
  { type: 'token', tokenId: 101, index: 0 },
  { type: 'token', tokenId: 202, index: 1 },
]);
assert.equal(events.at(-1).type, 'inspection-complete');
assert.equal(events.at(-1).receipt, streamedReceipt);
emitToken(999, '');
assert.equal(events.length, 3, 'Events after completion are ignored');
const unstreamedResult = streamHandle.inspect.generate('test');
releaseStream();
const unstreamedReceipt = await unstreamedResult;
assert.deepEqual(streamedReceipt.generationEvidence, unstreamedReceipt.generationEvidence);
assert.deepEqual(streamedReceipt.fingerprint, unstreamedReceipt.fingerprint);

const abortedEvents = [];
const streamController = new AbortController();
const abortedResult = streamHandle.inspect.generate('test', {
  generation: { signal: streamController.signal },
  onEvent: (event) => abortedEvents.push(event),
});
streamController.abort();
emitToken(999, '');
releaseStream();
await assert.rejects(abortedResult, { name: 'AbortError' });
assert.deepEqual(abortedEvents, [{ type: 'token', tokenId: 101, index: 0 }]);
await assert.rejects(streamHandle.inspect.generate('test', {
  onEvent() { throw new Error('Observer failure'); },
}), /Observer failure/);
assert.doesNotThrow(() => emitToken(999, ''), 'Failed runs close their event stream');
console.log('doppler-generation-evidence inspection streaming: ok');
