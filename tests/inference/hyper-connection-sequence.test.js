import assert from 'node:assert/strict';
import { test } from 'node:test';

// Real composite session and transaction owners; fake GPU command copies only.
// This exercises lifecycle, not shader arithmetic or model quality.
globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128, UNIFORM: 64 };
class Buffer {
  constructor({ size, usage, label }) { Object.assign(this, { size, usage, label, bytes: new Uint8Array(size) }); }
  destroy() { this.destroyed = true; }
}
globalThis.GPUBuffer = Buffer;
const { setDevice } = await import('../../src/gpu/device.js');
const { acquireBuffer, releaseBuffer, getBufferPool, destroyBufferPool } = await import('../../src/memory/buffer-pool.js');
const { HyperConnectionSession, createHyperConnectionSession } = await import('../../src/inference/pipelines/text/hyper-connection/session.js');
const { LAYER_PREFIXES } = await import('../../src/loader/layer-tensor-aliases.js');
const { NgramTokenHistory } = await import('../../src/inference/pipelines/text/ngram/ngram-addresses.js');
const { PleConvState } = await import('../../src/inference/pipelines/text/ngram/ple-layer.js');
const { withHyperConnectionTransaction, assertHyperConnectionLoad } = await import('../../src/inference/pipelines/text/hyper-connection/sequence-transaction.js');
const { InferencePipeline } = await import('../../src/inference/pipelines/text/pipeline.js');
const { PipelineGenerator } = await import('../../src/inference/pipelines/text/generator.js');

const device = {
  features: new Set(), limits: { maxStorageBufferBindingSize: 1 << 20, maxBufferSize: 1 << 20 },
  queue: {
    writeBuffer(buffer, offset, data) { buffer.bytes.set(new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength), offset); },
    submit(commands) { for (const command of commands) for (const copy of command) copy(); },
    onSubmittedWorkDone: async () => {},
  },
  createBindGroup() {},
  createBuffer: descriptor => new Buffer(descriptor),
  createCommandEncoder() {
    const copies = [];
    return { copyBufferToBuffer(source, sourceOffset, target, targetOffset, bytes) {
      copies.push(() => target.bytes.set(source.bytes.subarray(sourceOffset, sourceOffset + bytes), targetOffset));
    }, finish: () => copies };
  },
};
setDevice(device);
getBufferPool().configure({ enablePooling: false });

const hyper = { streamCount: 2, streamWidth: 2 };
const ngramLayer = { multipliers: [101n, 103n], headVocabSizes: [17], headOffsets: [0] };
const ngram = { semantics: 'qwen4_exp_ple', ngramSize: 2, headCount: 1, headsPerNgram: 1,
  unigramVocabSize: 20, boundaryTokenId: 19, layers: [ngramLayer] };

function fixture() {
  const history = new NgramTokenHistory(ngram, ngramLayer);
  history.append([1, 2]);
  const conv = new PleConvState({ channels: 4, stateLength: 2 });
  conv.buffer.bytes.fill(7);
  let qsaLength = 2;
  const qsa = { contract: { layers: [3] }, checkpoint: () => qsaLength,
    restore: value => { qsaLength = value; }, length: () => qsaLength, reset: () => { qsaLength = 0; }, destroy: async () => {} };
  const session = new HyperConnectionSession({ hyper, ngram, qsa, layers: new Map([[1, { history, conv, table: { clear() {} } }]]) });
  const linear = { seqLen: 2, convStateGPU: acquireBuffer(16), recurrentStateGPU: acquireBuffer(16) };
  linear.convStateGPU.bytes.fill(11); linear.recurrentStateGPU.bytes.fill(13);
  const state = { modelConfig: { hyperConnections: hyper }, hyperConnectionSession: session,
    linearAttentionRuntime: { schemaVersion: 1, layers: new Map([[0, linear]]) },
    currentSeqLen: 2, decodeStepCount: 2, isLoaded: true,
    kvCache: { length: 2, truncate(value) { this.length = value; } } };
  const mutate = () => {
    history.append([3]); conv.buffer.bytes.fill(99); qsaLength = 3; linear.seqLen = 3;
    linear.convStateGPU.bytes.fill(88); linear.recurrentStateGPU.bytes.fill(77);
    state.kvCache.length = state.currentSeqLen = 3; state.decodeStepCount = 3;
  };
  const unchanged = () => {
    assert.equal(state.currentSeqLen, 2); assert.equal(state.kvCache.length, 2);
    assert.equal(state.decodeStepCount, 2); assert.equal(linear.seqLen, 2);
    assert.equal(history.length, 2); assert.equal(qsaLength, 2);
    assert.ok(conv.buffer.bytes.every(value => value === 7));
    assert.ok(linear.convStateGPU.bytes.every(value => value === 11));
    assert.ok(linear.recurrentStateGPU.bytes.every(value => value === 13));
  };
  const close = async () => { await session.destroy(); releaseBuffer(linear.convStateGPU); releaseBuffer(linear.recurrentStateGPU); };
  return { state, session, history, mutate, unchanged, close };
}

await test('partial-layer failure restores KV, recurrent, indexer, n-gram and convolution state together; next call works', async () => {
  const f = fixture();
  try {
    const before = getBufferPool().getStats().activeBuffers;
    await assert.rejects(withHyperConnectionTransaction(f.state, {}, async () => { f.mutate(); throw Error('MoE failed'); }), /MoE failed/);
    f.unchanged();
    assert.equal(getBufferPool().getStats().activeBuffers, before);
    await withHyperConnectionTransaction(f.state, {}, async () => { f.mutate(); });
    assert.equal(f.state.currentSeqLen, 3); f.session.assertLength(3);
    assert.equal(getBufferPool().getStats().activeBuffers, before);
  } finally { await f.close(); }
});
await test('abort during the operation rolls back before allowing reuse', async () => {
  const f = fixture(); const controller = new AbortController();
  try {
    await assert.rejects(withHyperConnectionTransaction(f.state, { signal: controller.signal }, async () => {
      f.mutate(); controller.abort(); controller.signal.throwIfAborted();
    }), { name: 'AbortError' });
    f.unchanged();
  } finally { await f.close(); }
});
await test('nested prefill chunks share a transaction and a later chunk failure restores the initial prefix', async () => {
  const f = fixture();
  try {
    await assert.rejects(withHyperConnectionTransaction(f.state, {}, async opts => {
      await withHyperConnectionTransaction(f.state, { ...opts, _returnHidden: true }, async () => { f.mutate(); });
      throw Error('later chunk failed');
    }), /later chunk/);
    f.unchanged();
  } finally { await f.close(); }
});
await test('late cancellation releases a completed GPU result that was never transferred to the caller', async () => {
  const f = fixture(); const controller = new AbortController();
  try {
    const before = getBufferPool().getStats().activeBuffers;
    await assert.rejects(withHyperConnectionTransaction(f.state, { signal: controller.signal }, async () => {
      f.mutate(); controller.abort(); return { logitsBuffer: acquireBuffer(16) };
    }), { name: 'AbortError' });
    f.unchanged();
    assert.equal(getBufferPool().getStats().activeBuffers, before);
  } finally { await f.close(); }
});
await test('unsupported modes fail before mutation and snapshots fail before replacing KV', async () => {
  const f = fixture();
  try {
    await assert.rejects(withHyperConnectionTransaction(f.state, { useSpeculative: true }, async () => { f.mutate(); }), /not supported/);
    const cache = f.state.kvCache;
    assert.throws(() => InferencePipeline.prototype.applyKVCacheSnapshot.call(f.state, {}), /not supported/);
    const generator = new PipelineGenerator(f.state);
    assert.throws(() => generator.resetToSeqLen(0), /use resetGenerationState/);
    assert.throws(() => generator.resetToSeqLen(1), /not supported/);
    assert.equal(f.state.kvCache, cache); f.unchanged();
    const settings = { kvcache: { kvDtype: 'f16', layout: 'contiguous' }, decodeLoop: { batchSize: 1 } };
    const host = { ...f.state, useGPU: true, runtimeConfig: { inference: { compute: { activationDtype: 'f32' }, session: settings } } };
    assert.doesNotThrow(() => assertHyperConnectionLoad(host));
    settings.kvcache.kvDtype = 'f32';
    assert.doesNotThrow(() => assertHyperConnectionLoad(host));
    settings.kvcache.kvDtype = 'int8';
    assert.throws(() => assertHyperConnectionLoad(host), /contiguous f16\/f32/);
    settings.kvcache.kvDtype = 'f16';
    settings.decodeLoop.batchSize = 2;
    assert.throws(() => assertHyperConnectionLoad(host), /batchSize=1/);
  } finally { await f.close(); }
});
await test('closing waits for a pending row read and repeated closes share that completion', async () => {
  const f = fixture();
  const read = Promise.withResolvers();
  const conv = f.session.layers.get(1).conv;
  f.session.layers.get(1).table.gather = () => read.promise;
  const input = { shape: [1, 4] };
  const running = f.session.runPle(1, input, {}, { currentTokenIds: [3], currentSeqLen: 2 }, () => {});
  const rejected = assert.rejects(running, { name: 'AbortError' });
  let closed = false;
  const first = f.session.destroy();
  const second = f.session.destroy();
  assert.equal(first, second);
  first.then(() => { closed = true; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(closed, false); assert.notEqual(conv.buffer.destroyed, true);
  read.resolve(new Uint8Array(2));
  await Promise.all([first, rejected]);
  await f.close();
});

await test('partition sessions allocate only assigned QSA/PLE layers and isolate histories and cleanup', async () => {
  const layer = { ...ngramLayer, decoderLayerIndex: 1, rows: 17 };
  const contract = { ...ngram, headDim: 2, convStateLength: 2, tableDtype: 'bf16', layers: [layer] };
  const config = { numLayers: 8, hyperConnections: hyper,
    indexedAttention: { semantics: 'qwen4_exp_qsa', layers: [3, 7], headDim: 2, compressRatio: 4 },
    ngramEmbedding: contract, ngramEmbeddingSession: {
      rowCache: { maxRows: 4, maxBytes: 16 }, staging: { maxRows: 4 }, reads: { maxConcurrent: 1 },
    } };
  let reads = 0;
  const loader = { tensorLocations: new Map([[`${LAYER_PREFIXES(1)[0]}.ple.ple_embedding.ngram_embedding.shard_0.weight`,
    { role: 'ngram_table', dtype: 'BF16', shardIndex: 0, shape: [17, 2], offset: 0, size: 68 }]]),
    shardCache: { loadRange() { reads++; throw Error('Unrequested table read'); } } };
  const before = getBufferPool().getStats().activeBuffers;
  const a = await createHyperConnectionSession(config, loader, 8, [0, 3]);
  const b = await createHyperConnectionSession(config, loader, 8, [0, 3]);
  const c = await createHyperConnectionSession(config, loader, 8, [4, 7]);
  try {
    assert.deepEqual([...a.layers.keys()], [1]);
    assert.deepEqual([...a.qsa.layers.keys()], [3]);
    assert.deepEqual([...c.layers.keys()], []);
    assert.deepEqual([...c.qsa.layers.keys()], [7]);
    a.layers.get(1).history.append([2, 3]);
    assert.equal(b.layers.get(1).history.length, 0);
    a.layers.get(1).conv.buffer.bytes.fill(91);
    assert(b.layers.get(1).conv.buffer.bytes.every(v => v === 0));
    await a.destroy();
    assert.equal(b.layers.get(1).conv.destroyed, false);
    assert.equal(c.qsa.destroyed, false);
    assert.equal(reads, 0, 'Preparing an attempt never acquires table payloads');
    assert.deepEqual(config.indexedAttention.layers, [3, 7], 'Source contract remains immutable');
    await assert.rejects(createHyperConnectionSession(config, loader, 8, [4, 8]), /assigned layer range/);
  } finally { await Promise.all([a.destroy(), b.destroy(), c.destroy()]); }
  assert.equal(getBufferPool().getStats().activeBuffers, before);
});

await getBufferPool().settleDeferredDestruction();
assert.equal(getBufferPool().getStats().activeBuffers, 0);
destroyBufferPool(); setDevice(null);
