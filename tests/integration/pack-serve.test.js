import assert from 'node:assert/strict';
import http from 'node:http';
import { once, EventEmitter } from 'node:events';
import { createPackServeHandler } from '../../src/cli/serve/pack-handler.js';
import { writePackEvent } from '../../src/cli/serve/pack-http.js';
import { createDopplerRuntime } from '../../src/pack-runtime.js';
import { createModelHandle } from '../../src/client/runtime/model-session.js';
import { createPackProgramAdapter } from '../../src/client/runtime/pack-program-adapter.js';
import { hashPackObservation } from '../../src/config/pack-operation.js';
import { normalizePackServePolicy } from '../../src/config/pack-serve.js';
import { createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';

// Real HTTP -> public signed Pack session -> existing operation adapters.
// GPU/program components are synthetic, not physical model qualification.
const token = 'test-only-pack-serving-token';
const policy = { schema: 'doppler.pack-serve/v1', maxRequestBytes: 10000,
  maxOutputBytes: 100000, maxResponseBytes: 200000, maxDurationMs: 60000,
  allowedOrigins: ['https://application.example'] };
const generationOptions = { maxTokens: 2, maxSeqLen: 16, temperature: 0, topP: 1,
  topK: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 8, seed: 0, useChatTemplate: false };
const collect = async iterator => { const events = []; for await (const event of iterator) events.push(event); return events; };

function job(operation, application) {
  const inputs = {
    generate: { prompt: 'a public question' }, embed: { texts: ['first', 'second'], application },
    rerank: { query: 'a', documents: ['a', 'b'], application }, encodeSequence: { sequence: 'ACD' },
  };
  return { schema: 'doppler.pack-operation-request/v1', operation: { name: operation, version: 1 },
    input: inputs[operation], options: operation === 'generate' ? generationOptions
      : operation === 'encodeSequence' ? { includeLogits: false, includeTokenEmbeddings: false } : {},
    assignment: null, limits: { maxInputBytes: 10000, maxOutputBytes: 100000, deadlineAt: Date.now() + 30000 } };
}

async function openFixture(operation) {
  const fixture = await createSignedPackFixture({ operation, ...(operation === 'embed' ? { manifest: {
    modelId: 'pack-test-model', modelType: 'embedding', architecture: { hiddenSize: 4 },
    inference: { output: { embeddingPostprocessor: { poolingMode: 'last', includePrompt: true,
      projections: [], normalize: 'l2' } } },
  } } : {}) });
  let calls = 0;
  let closes = 0;
  let beforeRun = async () => {};
  let failGeneration = false;
  const gpu = { createBuffer: () => ({ destroy() {} }), createCommandEncoder() {}, queue: { writeBuffer() {} } };
  const program = {
    executionGraphHash: fixture.pack.program.executionGraphHash,
    tokenize() { return [1]; }, decodeTokens(ids) { return ids.join(','); },
    getTokenContract() { return { padTokenId: null, eosTokenId: null, stopTokenIds: [] }; },
    reset() {}, releaseStepResult() {}, async close() { closes += 1; },
    async executePhase() {
      calls += 1;
      if (failGeneration && calls > 1) throw Object.assign(new Error('Declared test execution failure.'), { code: 'TEST_EXECUTION_FAILURE' });
      return { logits: new Float32Array([0, 10]) };
    },
    async rerank({ options }) {
      calls += 1;
      await beforeRun(options.signal);
      return { schema: 'doppler_rerank_evidence/v1', inputHash: hashPackObservation('input'),
        outputHash: hashPackObservation('output'), backendIdentityHash: hashPackObservation('backend'), scores: [1, 0], ranking: [0, 1] };
    },
    async encodeSequence() { calls += 1; return { pooledEmbedding: new Float32Array([0.5, 1]), tokenEmbeddings: null, logits: null }; },
  };
  const runtime = createDopplerRuntime({
    device: { getDevice: () => gpu, getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
    artifactStore: fixture.artifactStore, trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
    async programFactory({ pack, targetPlan }) {
      if (operation !== 'embed') return program;
      const manifest = JSON.parse(new TextDecoder().decode(await fixture.artifactStore.readArtifact(pack.artifacts.find(a => a.artifactId === 'manifest'))));
      const pipeline = { manifest, isLoaded: true,
        resolvedRuntimeSession: { id: `sha256:${'1'.repeat(64)}` },
        getStats: () => ({ executionPlan: { transitions: [] } }),
        getKernelCapabilities: () => ({ adapterInfo: { vendor: 'synthetic' }, hasF16: false, hasSubgroups: false, maxBufferSize: 1024, deviceEpoch: 0 }),
        async embed() { calls += 1; return { embedding: new Float32Array([0, 0, 0, 1]), tokens: [1, 2], seqLen: 2, embeddingMode: 'last' }; },
        async unload() { closes += 1; },
      };
      const handle = createModelHandle(pipeline, { modelId: pack.modelId, manifestHash: pack.artifacts.find(a => a.artifactId === 'manifest').hash });
      return createPackProgramAdapter(handle, pack, targetPlan);
    },
  });
  return { session: await runtime.openPack(fixture.pack), fixture,
    get calls() { return calls; }, get closes() { return closes; },
    set beforeRun(value) { beforeRun = value; }, set failGeneration(value) { failGeneration = value; } };
}

async function serve(test, servingPolicy = policy) {
  const handler = createPackServeHandler({ session: test.session, token, policy: servingPolicy });
  const server = http.createServer(handler);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const base = `http://127.0.0.1:${server.address().port}`;
  return { handler, base, server,
    request: (body, options = {}) => fetch(`${base}${options.path ?? '/v1/operations'}`, {
      method: options.method ?? 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}`, ...options.headers },
      ...((options.method ?? 'POST') === 'POST' ? { body: typeof body === 'string' ? body : JSON.stringify(body) } : {}),
      signal: options.signal,
    }),
    async close() {
      await handler.close();
      await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    },
  };
}

for (const invalid of [null, {}, { ...policy, extra: true }, { ...policy, maxOutputBytes: null },
  { ...policy, maxRequestBytes: 0 }, { ...policy, maxDurationMs: 2147483648 },
  ...['*', 'null', 'https://application.example/path', 'https://user:secret@example.com', 4].map(origin => ({ ...policy, allowedOrigins: [origin] })),
  { ...policy, allowedOrigins: ['https://application.example', 'https://application.example'] }]) {
  assert.throws(() => normalizePackServePolicy(invalid));
}

for (const operation of ['generate', 'embed', 'rerank', 'encodeSequence']) {
  const test = await openFixture(operation);
  const server = await serve(test);
  try {
    const request = job(operation, test.fixture.pack.release.application);
    const expected = await collect(test.session.executeOperation(request));
    const response = await server.request(request);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), 'application/x-ndjson');
    const events = (await response.text()).trim().split('\n').map(line => JSON.parse(line));
    assert.deepEqual(events, expected, `${operation} retains the direct request/output/receipt/event identity`);
    assert.equal(events.at(-1).status, 'completed');
    const description = await server.request(null, { path: '/v1/model', method: 'GET' });
    assert.deepEqual(await description.json(), { modelId: test.session.modelId, pack: test.session.packIdentity,
      targetPlanDigest: test.session.selectedTargetPlanDigest, operations: [operation] });
  } finally { await server.close(); }
  assert.equal(test.session.closed, false, 'handler shutdown does not own the borrowed session');
  await test.session.close();
  assert.equal(test.closes, 1);
}

const test = await openFixture('rerank');
const mutablePolicy = structuredClone(policy);
const server = await serve(test, mutablePolicy);
mutablePolicy.maxRequestBytes = 1000000;
mutablePolicy.allowedOrigins.push('https://untrusted.example');
try {
  const request = job('rerank', test.fixture.pack.release.application);
  const rejection = async (body, status, code, options) => {
    const calls = test.calls;
    const response = await server.request(body, options);
    assert.equal(response.status, status);
    assert.equal((await response.json()).error.code, code);
    assert.equal(test.calls, calls, 'rejected request must not reach the model');
  };
  await rejection(request, 401, 'AUTH_REQUIRED', { headers: { Authorization: 'Bearer wrong' } });
  await rejection(request, 403, 'ORIGIN_DENIED', { headers: { Origin: 'https://untrusted.example' } });
  await rejection(request, 415, 'CONTENT_TYPE', { headers: { 'Content-Type': 'text/plain' } });
  await rejection(request, 404, 'NOT_FOUND', { path: '/v1/chat/completions' });
  await rejection('{', 400, 'INVALID_JSON');
  await rejection(' '.repeat(11000), 413, 'REQUEST_TOO_LARGE');
  await rejection({ ...request, operation: { name: 'customCode', version: 1 } }, 400, 'INVALID_OPERATION');
  await rejection({ ...request, assignment: { peer: 'someone' } }, 400, 'DELEGATION_UNSUPPORTED');
  await rejection({ ...request, limits: { ...request.limits, maxOutputBytes: policy.maxOutputBytes + 1 } }, 400, 'LIMIT_EXCEEDED');
  await rejection({ ...request, limits: { ...request.limits, deadlineAt: Date.now() + 120000 } }, 400, 'LIMIT_EXCEEDED');
  await rejection(job('generate'), 422, 'OPERATION_UNQUALIFIED');
  await rejection({ ...request, input: { ...request.input, application: {} } }, 500, 'PACK_EXECUTION_FAILED');
  const accepted = await server.request(request, { headers: { Origin: 'https://application.example' } });
  assert.equal(accepted.headers.get('access-control-allow-origin'), 'https://application.example');
  await accepted.text();
  const preflight = await server.request(null, { method: 'OPTIONS', headers: { Origin: 'https://application.example', Authorization: '' } });
  assert.equal(preflight.status, 204);

  // Body intake occupies the same bounded slot; an incomplete upload is cancellable.
  const stalled = http.request(`${server.base}/v1/operations`, { method: 'POST', headers: {
    Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', 'Content-Length': 100,
  } });
  const stalledResponse = once(stalled, 'response');
  const bodyReceived = once(server.server, 'request');
  stalled.write('{');
  await bodyReceived;
  const pending = await server.request(request);
  assert.equal(pending.status, 409);
  assert.equal((await pending.json()).error.code, 'SESSION_BUSY');
  await server.handler.close();
  const [stoppedResponse] = await stalledResponse;
  assert.equal(stoppedResponse.statusCode, 503);
  stoppedResponse.resume();
  stalled.end();
  await rejection(request, 503, 'SESSION_CLOSED');
} finally { await server.close(); await test.session.close(); }

// Client cancellation reaches the actual session, releases the slot, and preserves a later run.
const cancelled = await openFixture('rerank');
const cancellationServer = await serve(cancelled);
try {
  let started;
  let sawAbort;
  const startedPromise = new Promise(resolve => { started = resolve; });
  const abortedPromise = new Promise(resolve => { sawAbort = resolve; });
  cancelled.beforeRun = async signal => {
    started();
    await new Promise(resolve => signal.addEventListener('abort', () => { sawAbort(); resolve(); }, { once: true }));
    signal.throwIfAborted();
  };
  const controller = new AbortController();
  const response = cancellationServer.request(job('rerank', cancelled.fixture.pack.release.application), { signal: controller.signal });
  const rejected = assert.rejects(response, /abort/i);
  await startedPromise;
  const concurrent = await cancellationServer.request(job('rerank', cancelled.fixture.pack.release.application));
  assert.equal(concurrent.status, 409);
  await concurrent.text();
  controller.abort();
  await rejected;
  await abortedPromise;
  await cancellationServer.handler.close();
  assert.equal(cancelled.session.closed, false);
  cancelled.beforeRun = async () => {};
  assert.equal((await collect(cancelled.session.executeOperation(job('rerank', cancelled.fixture.pack.release.application)))).at(-1).status, 'completed');
} finally { await cancellationServer.close(); await cancelled.session.close(); }

// Stream failures are terminal error lines, never successful completion.
const failed = await openFixture('generate');
failed.failGeneration = true;
const failureServer = await serve(failed);
try {
  const response = await failureServer.request(job('generate'));
  const events = (await response.text()).trim().split('\n').map(line => JSON.parse(line));
  assert.equal(events[0].status, 'partial');
  assert.equal(events.at(-1).error.code, 'TEST_EXECUTION_FAILURE');
  assert.equal(events.some(event => event.status === 'completed'), false);
} finally { await failureServer.close(); await failed.session.close(); }

const bounded = await openFixture('generate');
const boundedServer = await serve(bounded, { ...policy, maxResponseBytes: 1 });
try {
  const response = await boundedServer.request(job('generate'));
  assert.equal(response.status, 413);
  assert.equal(await response.text(), '', 'even diagnostics cannot exceed the response ceiling');
} finally { await boundedServer.close(); await bounded.session.close(); }

const expired = await openFixture('rerank');
const deadlineServer = await serve(expired, { ...policy, maxDurationMs: 30 });
try {
  const stalled = http.request(`${deadlineServer.base}/v1/operations`, { method: 'POST', headers: {
    Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', 'Content-Length': 100,
  } });
  const responsePromise = once(stalled, 'response');
  stalled.write('{');
  const [response] = await responsePromise;
  assert.equal(response.statusCode, 408);
  response.resume();
  stalled.end();
  assert.equal(expired.calls, 0);
} finally { await deadlineServer.close(); await expired.session.close(); }

// Exercise deterministic transport backpressure without relying on OS socket buffers.
for (const abort of [false, true]) {
  const res = new EventEmitter();
  res.write = () => false;
  const controller = new AbortController();
  const write = writePackEvent(res, 'event\n', controller.signal);
  if (abort) { controller.abort(new Error('cancel blocked write')); await assert.rejects(write, /blocked write/); }
  else { res.emit('drain'); await write; }
  assert.equal(res.listenerCount('drain'), 0);
  assert.equal(res.listenerCount('error'), 0);
}
console.log('pack-serve.test: ok');
