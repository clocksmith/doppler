import assert from 'node:assert/strict';
import {
  DEFAULT_PREFIX_CACHE_POLICY,
  createServeHandler,
  createServePrefixCache,
  deriveServePrefixKey,
  parseServeArgs,
} from '../../src/cli/doppler-serve.js';

const mockTextModel = {
  modelId: 'qwen-3-5-2b-q4k-ehaf16',
  sourceCheckpointId: 'Qwen/Qwen3.5-2B',
  weightCapsuleId: 'qwen-3-5-2b-q4k-ehaf16-wp-catalog-v1',
  manifestVariantId: 'qwen-3-5-2b-q4k-ehaf16-mv-exec-v1',
  artifactCompleteness: 'complete',
  runtimePromotionState: 'manifest-owned',
  weightsRefAllowed: false,
  aliases: ['qwen35-2b'],
  modes: ['text'],
  hf: null,
};

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

function createMockRes({ destroyAfterWrite = null } = {}) {
  const state = {
    statusCode: null,
    headers: {},
    chunks: [],
    ended: false,
    headersSent: false,
    destroyed: false,
    writes: 0,
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
      state.writes += 1;
      if (destroyAfterWrite !== null && state.writes >= destroyAfterWrite) {
        state.destroyed = true;
      }
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

// 1. CLI args parsing for prefix cache
{
  const defaults = parseServeArgs([]);
  assert.equal(defaults.prefixCache, true);
  assert.equal(defaults.prefixCacheBytes, DEFAULT_PREFIX_CACHE_POLICY.maxBytes);

  const disabled = parseServeArgs(['--no-prefix-cache']);
  assert.equal(disabled.prefixCache, false);

  // Non-loopback binds disable default-on prefix caching unless explicitly opted in.
  const nonLoopback = parseServeArgs(['--host', '0.0.0.0']);
  assert.equal(nonLoopback.prefixCache, false);
  const nonLoopbackLan = parseServeArgs(['--host', '192.168.1.10']);
  assert.equal(nonLoopbackLan.prefixCache, false);
  const nonLoopbackOptIn = parseServeArgs(['--host', '0.0.0.0', '--prefix-cache']);
  assert.equal(nonLoopbackOptIn.prefixCache, true);
  const ipv6Loopback = parseServeArgs(['--host', '::1']);
  assert.equal(ipv6Loopback.prefixCache, true);
  const localhostBind = parseServeArgs(['--host', 'localhost']);
  assert.equal(localhostBind.prefixCache, true);

  const custom = parseServeArgs(['--prefix-cache-bytes', '67108864']);
  assert.equal(custom.prefixCache, true);
  assert.equal(custom.prefixCacheBytes, 67108864);

  assert.throws(() => parseServeArgs(['--prefix-cache-bytes', '0']), /positive integer/);
  assert.throws(() => parseServeArgs(['--prefix-cache-bytes', 'abc']), /positive integer/);
  assert.throws(() => parseServeArgs(['--prefix-cache-bytes']), /Missing value/);
}

// 2. Deterministic multi-turn key derivation
{
  const turn1 = [
    { role: 'system', content: 'You are concise.' },
    { role: 'user', content: 'Explain WebGPU.' },
  ];
  const turn2 = [
    ...turn1,
    { role: 'assistant', content: 'WebGPU is a modern GPU API.' },
    { role: 'user', content: 'How does it compare to Vulkan?' },
  ];
  const otherConv = [
    { role: 'system', content: 'You are concise.' },
    { role: 'user', content: 'Explain SQLite.' },
  ];

  const k1 = deriveServePrefixKey('model-a', turn1);
  const k2 = deriveServePrefixKey('model-a', turn2);
  const kOther = deriveServePrefixKey('model-a', otherConv);
  const kDiffModel = deriveServePrefixKey('model-b', turn1);

  assert.ok(k1 && k1.startsWith('prefix_'));
  assert.equal(k1, k2, 'Multi-turn follow-up must reuse the same prefix key');
  assert.notEqual(k1, kOther, 'Distinct conversations must not alias');
  assert.notEqual(k1, kDiffModel, 'Distinct models must not alias');
}

// 3. LRU eviction, byte caps, busy skip, discard-on-failure, and idle TTL
{
  let clock = 1_000;
  let nextId = 0;
  const closed = [];
  const makeCtx = (bytes) => ({
    id: ++nextId,
    closed: false,
    bytes,
    getStats() {
      return { allocatedBytes: this.bytes, cachedTokens: 128, reusedTokens: 64 };
    },
    close() {
      this.closed = true;
      closed.push(this.id);
    },
  });

  const cache = createServePrefixCache({
    policy: { maxEntries: 2, maxBytes: 200, maxEntryBytes: 100, idleMs: 500 },
    now: () => clock,
  });

  const l1 = cache.acquire('k1', () => makeCtx(90));
  assert.equal(l1.hit, false);
  // Concurrent acquire on active key degrades cleanly to uncached null
  const concurrent = cache.acquire('k1', () => makeCtx(90));
  assert.equal(concurrent, null);
  assert.equal(cache.getStats().busySkips, 1);
  l1.release();

  // Hit on k1 reuses same context
  const l1Hit = cache.acquire('k1', () => makeCtx(90));
  assert.equal(l1Hit.hit, true);
  assert.equal(l1Hit.context.id, l1.context.id);
  l1Hit.release();

  // k2 fills budget (90 + 100 <= 200)
  const l2 = cache.acquire('k2', () => makeCtx(90));
  assert.equal(l2.hit, false);
  l2.release();

  // Touch k1 so k2 becomes LRU victim, then insert k3 -> evicts k2
  const l1Touch = cache.acquire('k1', () => makeCtx(90));
  l1Touch.release();
  const l3 = cache.acquire('k3', () => makeCtx(90));
  assert.equal(l3.hit, false);
  assert.deepEqual(closed, [2]);
  assert.equal(cache.getStats().evictions, 1);

  // Discard on error closes k3 immediately
  l3.release({ discard: true });
  assert.deepEqual(closed, [2, 3]);
  assert.equal(cache.getStats().entries, 1);

  // Advance past idleMs -> sweep evicts k1
  clock += 600;
  const l4 = cache.acquire('k4', () => makeCtx(90));
  assert.deepEqual(closed, [2, 3, 1]);
  l4.release();
  cache.close();
  assert.deepEqual(closed, [2, 3, 1, 4]);
}

// 4. End-to-end handler integration: non-streaming & streaming reuse + cached_tokens + abort discard
{
  const createdContexts = [];
  const chatCalls = [];
  const mockModelHandle = {
    supportsGenerationContext: true,
    createGenerationContext({ maxBytes }) {
      const ctx = {
        id: createdContexts.length + 1,
        maxBytes,
        closed: false,
        turns: 0,
        getStats() {
          return {
            allocatedBytes: 4096,
            cachedTokens: this.turns * 32,
            reusedTokens: this.turns > 1 ? 32 : 0,
          };
        },
        close() {
          this.closed = true;
        },
      };
      createdContexts.push(ctx);
      return ctx;
    },
    async chatText(messages, options) {
      const ctx = options.generationContext ?? null;
      const reused = ctx && ctx.turns > 0 ? 32 : 0;
      if (ctx) ctx.turns += 1;
      chatCalls.push({ kind: 'chatText', contextId: ctx?.id ?? null, messages });
      return {
        content: `reply-${chatCalls.length}`,
        usage: { promptTokens: 48, completionTokens: 8, totalTokens: 56 },
        evidence: { stats: { reusedPrefixTokens: reused } },
      };
    },
    async *chat(messages, options) {
      const ctx = options.generationContext ?? null;
      if (ctx) ctx.turns += 1;
      chatCalls.push({ kind: 'chat', contextId: ctx?.id ?? null, messages });
      yield 'tok1';
      yield 'tok2';
    },
  };

  const prefixCache = createServePrefixCache();
  const handler = createServeHandler({
    prefixCache,
    listModels: async () => [mockTextModel],
    resolveModel: async () => mockTextModel,
    getCachedModel: async () => mockModelHandle,
    dopplerClient: {
      async chatText() {
        throw new Error('Should call loaded model handle when prefix cache is enabled');
      },
      async *chat() {
        throw new Error('Should call loaded model handle when prefix cache is enabled');
      },
    },
  });

  const turn1 = [
    { role: 'system', content: 'System prompt.' },
    { role: 'user', content: 'First question.' },
  ];
  const res1 = createMockRes();
  await handler(createMockReq('POST', '/v1/chat/completions', { model: 'qwen35-2b', messages: turn1 }), res1);
  assert.equal(res1.state.statusCode, 200);
  const body1 = parseBody(res1);
  assert.equal(body1.usage.prompt_tokens_details, undefined);

  const turn2 = [
    ...turn1,
    { role: 'assistant', content: 'reply-1' },
    { role: 'user', content: 'Follow-up question.' },
  ];
  const res2 = createMockRes();
  await handler(createMockReq('POST', '/v1/chat/completions', { model: 'qwen35-2b', messages: turn2 }), res2);
  assert.equal(res2.state.statusCode, 200);
  const body2 = parseBody(res2);
  assert.deepEqual(body2.usage.prompt_tokens_details, { cached_tokens: 32 });
  assert.equal(createdContexts.length, 1);
  assert.equal(chatCalls[0].contextId, 1);
  assert.equal(chatCalls[1].contextId, 1);

  // Streaming turn 3 hits same context
  const turn3 = [
    ...turn2,
    { role: 'assistant', content: 'reply-2' },
    { role: 'user', content: 'Third question.' },
  ];
  const res3 = createMockRes();
  await handler(createMockReq('POST', '/v1/chat/completions', { model: 'qwen35-2b', messages: turn3, stream: true }), res3);
  assert.equal(res3.state.statusCode, 200);
  assert.equal(chatCalls[2].contextId, 1);
  assert.equal(createdContexts[0].closed, false);

  // Health reports prefix_cache stats
  const healthRes = createMockRes();
  await handler(createMockReq('GET', '/health'), healthRes);
  const health = parseBody(healthRes);
  assert.equal(health.prefix_cache.enabled, true);
  assert.equal(health.prefix_cache.hits, 2);
  assert.equal(health.prefix_cache.misses, 1);
  assert.equal(health.prefix_cache.entries, 1);

  // Aborted streaming response discards its context cleanly
  const resAbort = createMockRes({ destroyAfterWrite: 2 });
  await handler(createMockReq('POST', '/v1/chat/completions', { model: 'qwen35-2b', messages: turn3, stream: true }), resAbort);
  assert.equal(createdContexts[0].closed, true);
  assert.equal(prefixCache.getStats().entries, 0);

  prefixCache.close();
}



// An embedded handler has no knowledge of the socket it will listen on.
{
  const health = createMockRes();
  await createServeHandler()(createMockReq('GET', '/health'), health);
  assert.equal(parseBody(health).prefix_cache.enabled, false);
}

// Reloading the same model creates a different owner even with identical prompt tokens.
{
  const cache = createServePrefixCache();
  const owners = [{}, {}];
  let closed = 0;
  const make = () => ({
    close() { closed++; },
    getStats() { return { allocatedBytes: 8, cachedTokens: 1, reusedTokens: 0, status: 'cached' }; },
  });
  const first = cache.acquire('root', make, owners[0]);
  first.release();
  const second = cache.acquire('root', make, owners[1]);
  assert.equal(second.hit, false);
  assert.notEqual(first.context, second.context);
  assert.equal(closed, 1);
  second.release();
  cache.close();
}

// A successful answer can still exceed the KV budget or finish without a checkpoint.
// Looking up that conversation again must be a miss, even on the same model handle.
{
  const cache = createServePrefixCache();
  let closed = 0;
  const owner = {};
  const initial = cache.getStats();
  for (const status of ['budget', 'discarded', 'empty']) {
    const lease = cache.acquire('same-turn', () => ({
      getStats: () => ({ allocatedBytes: 0, cachedTokens: 0, reusedTokens: 0, status }),
      close() { closed++; },
    }), owner);
    assert.equal(lease.hit, false);
    lease.release();
    lease.release();
    assert.equal(cache.getStats().entries, 0);
  }
  assert.deepEqual(initial.unretained, { budget: 0, discarded: 0, empty: 0 });
  const stats = cache.getStats();
  assert.deepEqual(stats.unretained, { budget: 1, discarded: 1, empty: 1 });
  assert.equal(stats.hits, 0);
  assert.equal(stats.misses, 3);
  assert.equal(stats.evictions, 0);
  assert.equal(closed, 3);
  cache.close();
}

console.log('doppler-serve-prefix-cache.test: ok');
