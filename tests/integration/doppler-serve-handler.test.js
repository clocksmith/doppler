import assert from 'node:assert/strict';
import { createServeHandler } from '../../src/cli/doppler-serve.js';

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

function createMockHandler() {
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
  });
  return { handler: mockHandler, calls };
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
  assert.equal(body.doppler_receipt.requestedModel, 'gemma3-270m');
  assert.equal(body.doppler_receipt.resolvedModel, mockTextModel.modelId);
  assert.equal(body.doppler_receipt.artifact.weightPackId, mockTextModel.weightPackId);
  assert.equal(body.doppler_receipt.artifact.hf.repoId, 'Clocksmith/rdrr');
  assert.equal(body.doppler_receipt.generation.maxTokens, 8);
  assert.equal(body.doppler_receipt.generation.temperature, 0);
  assert.equal(body.doppler_receipt.generation.topK, 1);
  assert.deepEqual(body.doppler_receipt.usage, {
    promptTokens: 4,
    completionTokens: 3,
    totalTokens: 7,
  });
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
