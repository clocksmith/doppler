import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createServeHandler, parseServeArgs } from '../../src/cli/doppler-serve.js';

const handler = createServeHandler();
const mockTextModel = {
  modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
  sourceCheckpointId: 'google/gemma-3-270m-it',
  weightPackId: 'gemma-3-270m-it-q4k-ehf16-af32-wp-catalog-v1',
  manifestVariantId: 'gemma-3-270m-it-q4k-ehf16-af32-mv-exec-v1',
  artifactCompleteness: 'complete',
  runtimePromotionState: 'manifest-owned',
  weightsRefAllowed: false,
  aliases: ['gemma3-270m'],
  modes: ['text', 'vision'],
  hf: {
    repoId: 'Clocksmith/rdrr',
    revision: 'abc123',
    path: 'models/gemma-3-270m-it-q4k-ehf16-af32',
  },
};
const mockEmbeddingModel = {
  modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
  sourceCheckpointId: 'google/embeddinggemma-300m',
  weightPackId: 'google-embeddinggemma-300m-q4k-ehf16-af32-wp-catalog-v1',
  manifestVariantId: 'google-embeddinggemma-300m-q4k-ehf16-af32-mv-exec-v1',
  artifactCompleteness: 'complete',
  runtimePromotionState: 'manifest-owned',
  weightsRefAllowed: false,
  aliases: ['embeddinggemma-300m'],
  modes: ['embedding'],
  hf: {
    repoId: 'Clocksmith/rdrr',
    revision: 'def456',
    path: 'models/google-embeddinggemma-300m-q4k-ehf16-af32',
  },
};

function createMockHandler(overrides = {}) {
  const calls = [];
  const mockHandler = createServeHandler({
    listModels: async () => [mockTextModel, mockEmbeddingModel],
    resolveModel: async (model) => {
      if (model === mockTextModel.modelId || mockTextModel.aliases.includes(model)) {
        return mockTextModel;
      }
      if (model === mockEmbeddingModel.modelId || mockEmbeddingModel.aliases.includes(model)) {
        return mockEmbeddingModel;
      }
      throw new Error(`Unknown quickstart model "${model}".`);
    },
    dopplerClient: {
      async chatText(messages, options) {
        calls.push({ kind: 'chatText', messages, options });
        return {
          content: 'Hello from Doppler.',
          usage: {
            promptTokens: 4,
            completionTokens: 3,
            totalTokens: 7,
          },
        };
      },
      async *chat(messages, options) {
        calls.push({ kind: 'chat', messages, options });
        yield 'Hello';
      },
    },
    ...overrides,
  });
  return { handler: mockHandler, calls };
}

function createFailingMockHandler(error) {
  const mockHandler = createServeHandler({
    listModels: async () => [mockTextModel],
    resolveModel: async (model) => {
      if (model === mockTextModel.modelId || mockTextModel.aliases.includes(model)) {
        return mockTextModel;
      }
      throw new Error(`Unknown quickstart model "${model}".`);
    },
    dopplerClient: {
      async chatText() {
        throw error;
      },
      async *chat() {
        throw error;
      },
    },
  });
  return mockHandler;
}

function createMockReq(method, url, body) {
  const bodyStr = body != null ? JSON.stringify(body) : '';
  const chunks = bodyStr ? [Buffer.from(bodyStr)] : [];
  let consumed = false;
  return {
    method,
    url,
    headers: { host: 'localhost:8080', 'content-type': 'application/json' },
    [Symbol.asyncIterator]() {
      return {
        next() {
          if (!consumed) {
            consumed = true;
            if (chunks.length > 0) {
              return Promise.resolve({ value: chunks[0], done: false });
            }
          }
          return Promise.resolve({ value: undefined, done: true });
        },
      };
    },
  };
}

function createMockRes() {
  const state = {
    statusCode: null,
    headers: {},
    chunks: [],
    ended: false,
    headersSent: false,
    destroyed: false,
  };
  return {
    state,
    writeHead(code, headers) {
      state.statusCode = code;
      state.headers = { ...state.headers, ...headers };
      state.headersSent = true;
    },
    write(chunk) {
      state.chunks.push(chunk);
    },
    end(body) {
      if (body) state.chunks.push(body);
      state.ended = true;
    },
    get headersSent() {
      return state.headersSent;
    },
    get destroyed() {
      return state.destroyed;
    },
  };
}

function parseBody(res) {
  return JSON.parse(res.state.chunks.join(''));
}

function sha256Hex(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function createLoadFailure() {
  const error = new Error(
    'Pipeline load phase "loadWeights" failed: Storage buffer size 805306368 exceeds device maxStorageBufferBindingSize (134217728).'
  );
  error.details = {
    pipelineLoadPhase: 'loadWeights',
    modelId: mockTextModel.modelId,
    weightLoadFailure: {
      tensorName: 'model.language_model.embed_tokens.weight',
      tensorRole: 'embedding',
      tensorDtype: 'F16',
      tensorShape: [262144, 1536],
      tensorSizeBytes: 805306368,
      tensorLoadStage: 'streamShardToGpuBuffer',
      toGPU: true,
      streamedUpload: true,
      deviceLimitFailure: {
        kind: 'gpu_resident_embedding_exceeds_device_limit',
        maxGpuResidentBytes: 134217728,
        maxStorageBufferBindingSize: 134217728,
        maxBufferSize: 134217728,
        maxStorageBuffersPerShaderStage: null,
        largeWeightMaxBytes: 107374182,
        embeddingKernel: {
          kernel: 'gather.wgsl',
          entry: 'main',
        },
        splitKernelExpected: false,
        activeSplitKernelMaxSections: null,
        maxSplitEmbeddingSections: 8,
        requiredSplitSections: 8,
      },
    },
  };
  return error;
}

// CLI args - model URL requires an explicit model identity
{
  assert.throws(
    () => parseServeArgs(['--model-url', 'models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple']),
    /--model-url requires --model/
  );
}

// CLI args - local model path is normalized to file URL
{
  const localPath = 'models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple';
  const settings = parseServeArgs(['--model', 'gemma4-e2b-int4ple', '--model-url', localPath]);
  assert.equal(settings.model, 'gemma4-e2b-int4ple');
  assert.equal(settings.modelUrl, pathToFileURL(path.resolve(localPath)).href);
}

// Health endpoint
{
  const req = createMockReq('GET', '/health');
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 200);
  const body = parseBody(res);
  assert.equal(body.status, 'ok');
  assert.equal(typeof body.version, 'string');
}

// Root serves health
{
  const req = createMockReq('GET', '/');
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 200);
  const body = parseBody(res);
  assert.equal(body.status, 'ok');
}

// Models endpoint
{
  const req = createMockReq('GET', '/v1/models');
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 200);
  const body = parseBody(res);
  assert.equal(body.object, 'list');
  assert.ok(Array.isArray(body.data));
  assert.ok(body.data.length >= 4, `Expected at least 4 text models, got ${body.data.length}`);
  const modelIds = body.data.map((m) => m.id);
  assert.ok(modelIds.includes('gemma-3-270m-it-q4k-ehf16-af32'));
  assert.ok(modelIds.includes('gemma-3-1b-it-q4k-ehf16-af32'));
  assert.ok(modelIds.includes('gemma-4-e2b-it-q4k-ehf16-af32'));
  assert.ok(modelIds.includes('gemma-4-e2b-it-q4k-ehf16-af32-int4ple'));
  assert.ok(!modelIds.includes('qwen-3-5-0-8b-q4k-ehaf16'));
  assert.ok(!modelIds.includes('qwen-3-5-2b-q4k-ehaf16'));
  for (const entry of body.data) {
    assert.equal(entry.object, 'model');
    assert.equal(entry.owned_by, 'doppler');
    assert.ok(entry.doppler);
    assert.equal(entry.doppler.artifactCompleteness, 'complete');
    assert.equal(entry.doppler.runtimePromotionState, 'manifest-owned');
    assert.equal(entry.doppler.weightsRefAllowed, false);
    assert.ok(entry.doppler.sourceCheckpointId);
    assert.ok(entry.doppler.weightPackId);
    assert.ok(entry.doppler.manifestVariantId);
    assert.ok(entry.doppler.modes.includes('text'));
  }
}

// 404 for unknown endpoint
{
  const req = createMockReq('GET', '/v1/unknown');
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 404);
  const body = parseBody(res);
  assert.ok(body.error);
  assert.equal(body.error.type, 'not_found');
}

// Chat completions - missing body
{
  const req = createMockReq('POST', '/v1/chat/completions');
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 400);
  const body = parseBody(res);
  assert.ok(body.error);
}

// Chat completions - missing model
{
  const req = createMockReq('POST', '/v1/chat/completions', {
    messages: [{ role: 'user', content: 'hi' }],
  });
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 400);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('model'));
}

// Chat completions - missing messages
{
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'gemma3-270m',
  });
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 400);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('messages'));
}

// Chat completions - invalid messages
{
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'gemma3-270m',
    messages: [{ content: 'hi' }],
  });
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 400);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('role'));
}

// Chat completions - unknown model fails before runtime load
{
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'nonexistent-model',
    messages: [{ role: 'user', content: 'hi' }],
  });
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 400);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('Unknown model'));
}

// Chat completions - embedding-only models are not exposed as chat models
{
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'embeddinggemma-300m',
    messages: [{ role: 'user', content: 'hi' }],
  });
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 400);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('not text-generative'));
}

// Chat completions - explicit runtime model source keeps response identity canonical
{
  const runtimeModel = { url: 'file:///models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/' };
  const runtimeCalls = [];
  const mock = createMockHandler({
    resolveRuntimeModel(registryEntry, requestedModel) {
      runtimeCalls.push({ registryEntry, requestedModel });
      return runtimeModel;
    },
  });
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'gemma3-270m',
    messages: [{ role: 'user', content: 'hi' }],
    include_receipt: true,
  });
  const res = createMockRes();
  await mock.handler(req, res);
  assert.equal(res.state.statusCode, 200);
  assert.equal(runtimeCalls.length, 1);
  assert.equal(runtimeCalls[0].registryEntry.modelId, mockTextModel.modelId);
  assert.equal(runtimeCalls[0].requestedModel, 'gemma3-270m');
  assert.equal(mock.calls.length, 1);
  assert.deepEqual(mock.calls[0].options.model, runtimeModel);
  const body = parseBody(res);
  assert.equal(body.model, mockTextModel.modelId);
  assert.equal(body.doppler_receipt.requestedModel, 'gemma3-270m');
  assert.equal(body.doppler_receipt.resolvedModel, mockTextModel.modelId);
  assert.deepEqual(body.doppler_receipt.runtimeModelSource, {
    kind: 'url',
    url: runtimeModel.url,
  });
  assert.equal(body.doppler_receipt.artifact.source, 'quickstart-registry');
}

// Chat completions - streaming uses explicit runtime model source too
{
  const runtimeModel = { url: 'file:///models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/' };
  const runtimeCalls = [];
  const mock = createMockHandler({
    resolveRuntimeModel(registryEntry, requestedModel) {
      runtimeCalls.push({ registryEntry, requestedModel });
      return runtimeModel;
    },
  });
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'gemma3-270m',
    messages: [{ role: 'user', content: 'hi' }],
    stream: true,
  });
  const res = createMockRes();
  await mock.handler(req, res);
  assert.equal(res.state.statusCode, 200);
  assert.equal(runtimeCalls.length, 1);
  assert.equal(runtimeCalls[0].registryEntry.modelId, mockTextModel.modelId);
  assert.equal(runtimeCalls[0].requestedModel, 'gemma3-270m');
  assert.equal(mock.calls.length, 1);
  assert.deepEqual(mock.calls[0].options.model, runtimeModel);
  const streamText = res.state.chunks.join('');
  assert.ok(streamText.includes(`"model":"${mockTextModel.modelId}"`));
  assert.ok(streamText.includes('data: [DONE]'));
}

// Chat completions - optional Doppler receipt
{
  const mock = createMockHandler();
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'gemma3-270m',
    messages: [{ role: 'user', content: 'hi' }],
    max_tokens: 8,
    temperature: 0,
    top_k: 1,
    include_receipt: true,
  });
  const res = createMockRes();
  await mock.handler(req, res);
  assert.equal(res.state.statusCode, 200);
  assert.equal(mock.calls.length, 1);
  assert.equal(mock.calls[0].options.model, mockTextModel.modelId);
  assert.equal(mock.calls[0].options.maxTokens, 8);
  const body = parseBody(res);
  assert.equal(body.model, mockTextModel.modelId);
  assert.equal(body.choices[0].message.content, 'Hello from Doppler.');
  assert.equal(body.usage.total_tokens, 7);
  assert.equal(body.doppler_receipt.receiptVersion, 'doppler_serve_receipt_v1');
  assert.equal(body.doppler_receipt.schemaVersion, 1);
  assert.equal(body.doppler_receipt.surface, 'serve');
  assert.equal(body.doppler_receipt.endpoint, '/v1/chat/completions');
  assert.equal(body.doppler_receipt.status, 'pass');
  assert.equal(body.doppler_receipt.runtime, 'doppler-gpu');
  assert.equal(body.doppler_receipt.runtimePath, 'doppler-gpu.chatText');
  assert.deepEqual(body.doppler_receipt.runtimeModelSource, {
    kind: 'quickstart-registry',
    modelId: mockTextModel.modelId,
  });
  assert.equal(body.doppler_receipt.modelId, mockTextModel.modelId);
  assert.equal(body.doppler_receipt.requestedModel, 'gemma3-270m');
  assert.equal(body.doppler_receipt.resolvedModel, mockTextModel.modelId);
  assert.equal(body.doppler_receipt.artifact.weightPackId, mockTextModel.weightPackId);
  assert.equal(body.doppler_receipt.artifact.hf.repoId, 'Clocksmith/rdrr');
  assert.equal(body.doppler_receipt.request.messages.count, 1);
  assert.equal(body.doppler_receipt.request.messages.digest.algorithm, 'sha256');
  assert.match(body.doppler_receipt.request.messages.digest.value, /^[0-9a-f]{64}$/);
  assert.equal(body.doppler_receipt.request.messages.digest.bytes, 32);
  assert.equal(body.doppler_receipt.request.generationDigest.algorithm, 'sha256');
  assert.match(body.doppler_receipt.request.generationDigest.value, /^[0-9a-f]{64}$/);
  assert.equal(body.doppler_receipt.output.role, 'assistant');
  assert.equal(body.doppler_receipt.output.digest.value, sha256Hex('Hello from Doppler.'));
  assert.equal(body.doppler_receipt.output.digest.bytes, 19);
  assert.equal(body.doppler_receipt.output.textLength, 19);
  assert.equal(body.doppler_receipt.output.empty, false);
  assert.equal(body.doppler_receipt.transcript.digest.algorithm, 'sha256');
  assert.match(body.doppler_receipt.transcript.digest.value, /^[0-9a-f]{64}$/);
  assert.equal(body.doppler_receipt.generation.maxTokens, 8);
  assert.equal(body.doppler_receipt.generation.temperature, 0);
  assert.equal(body.doppler_receipt.generation.topK, 1);
  assert.deepEqual(body.doppler_receipt.usage, {
    promptTokens: 4,
    completionTokens: 3,
    totalTokens: 7,
  });
}

// Chat completions - failed receipt keeps runtime blocker attributable
{
  const failingHandler = createFailingMockHandler(createLoadFailure());
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'gemma3-270m',
    messages: [{ role: 'user', content: 'hi' }],
    max_tokens: 8,
    temperature: 0,
    top_k: 1,
    include_receipt: true,
  });
  const res = createMockRes();
  await failingHandler(req, res);
  assert.equal(res.state.statusCode, 500);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('loadWeights'));
  assert.equal(body.doppler_receipt.receiptVersion, 'doppler_serve_receipt_v1');
  assert.equal(body.doppler_receipt.surface, 'serve');
  assert.equal(body.doppler_receipt.status, 'diagnostic');
  assert.equal(body.doppler_receipt.runtimePath, 'doppler-gpu.chatText');
  assert.equal(body.doppler_receipt.modelId, mockTextModel.modelId);
  assert.equal(body.doppler_receipt.request.messages.count, 1);
  assert.equal(body.doppler_receipt.request.messages.digest.algorithm, 'sha256');
  assert.match(body.doppler_receipt.request.messages.digest.value, /^[0-9a-f]{64}$/);
  assert.equal(body.doppler_receipt.generation.maxTokens, 8);
  assert.equal(body.doppler_receipt.failure.code, 'pipeline-load-failed');
  assert.equal(body.doppler_receipt.failure.stage, 'loadWeights');
  assert.equal(body.doppler_receipt.failure.modelId, mockTextModel.modelId);
  assert.equal(body.doppler_receipt.failure.weightLoadFailure.tensorRole, 'embedding');
  assert.equal(body.doppler_receipt.failure.weightLoadFailure.tensorSizeBytes, 805306368);
  assert.equal(
    body.doppler_receipt.failure.weightLoadFailure.deviceLimitFailure.kind,
    'gpu_resident_embedding_exceeds_device_limit'
  );
  assert.equal(
    body.doppler_receipt.failure.weightLoadFailure.deviceLimitFailure.maxGpuResidentBytes,
    134217728
  );
  assert.deepEqual(
    body.doppler_receipt.failure.weightLoadFailure.deviceLimitFailure.embeddingKernel,
    {
      kernel: 'gather.wgsl',
      entry: 'main',
    }
  );
}

// Chat completions - receipt is explicit and unavailable on stream responses
{
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'gemma3-270m',
    messages: [{ role: 'user', content: 'hi' }],
    stream: true,
    include_receipt: true,
  });
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 400);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('include_receipt'));
}

// Chat completions - receipt request aliases must agree
{
  const req = createMockReq('POST', '/v1/chat/completions', {
    model: 'gemma3-270m',
    messages: [{ role: 'user', content: 'hi' }],
    include_receipt: false,
    doppler_receipt: true,
  });
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 400);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('must agree'));
}

// CORS preflight
{
  const req = createMockReq('OPTIONS', '/v1/chat/completions');
  const res = createMockRes();
  await handler(req, res);
  assert.equal(res.state.statusCode, 204);
  assert.equal(res.state.headers['Access-Control-Allow-Origin'], '*');
  assert.ok(res.state.headers['Access-Control-Allow-Methods'].includes('POST'));
}

console.log('doppler-serve-handler.test: ok');
