import assert from 'node:assert/strict';
import { createServeHandler } from '../../src/cli/doppler-serve.js';

const handler = createServeHandler();

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
  assert.ok(modelIds.includes('qwen-3-5-0-8b-q4k-ehaf16'));
  assert.ok(modelIds.includes('qwen-3-5-2b-q4k-ehaf16'));
  for (const entry of body.data) {
    assert.equal(entry.object, 'model');
    assert.equal(entry.owned_by, 'doppler');
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
  assert.equal(res.state.statusCode, 500);
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
  assert.equal(res.state.statusCode, 500);
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
  assert.equal(res.state.statusCode, 500);
  const body = parseBody(res);
  assert.ok(body.error.message.includes('role'));
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
