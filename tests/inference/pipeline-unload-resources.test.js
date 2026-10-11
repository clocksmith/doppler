import assert from 'node:assert/strict';
import { test } from 'node:test';
import { InferencePipeline } from '../../src/inference/pipelines/text.js';
import { KVCache } from '../../src/inference/kv-cache/base.js';
import { DecodeBufferManager } from '../../src/inference/decode-buffers.js';
import { setDevice } from '../../src/gpu/device.js';

function fixture() {
  const pipeline = new InferencePipeline();
  const buffers = [];
  const create = () => {
    const buffer = { destroys: 0, destroy() { this.destroys++; } };
    buffers.push(buffer); return buffer;
  };
  const cache = new KVCache({ numLayers: 2, numHeads: 1, headDim: 2,
    maxSeqLen: 4, useGPU: false, layout: 'contiguous', pageSize: 4, kvDtype: 'f32' });
  for (const layer of cache.layers) { layer.keysGPU = create(); layer.valuesGPU = create(); }
  pipeline.kvCache = cache;
  pipeline.decodeBuffers.buffers = {
    hidden: create(), hiddenAlt: create(), attnOutput: create(), ffnIntermediate: create(),
  };
  pipeline.isLoaded = true;
  return { pipeline, buffers };
}

test('unload destroys session KV and decode allocations, not another session', async () => {
  const first = fixture(), second = fixture();
  await first.pipeline.unload();
  assert(first.buffers.every(buffer => buffer.destroys === 1), 'clear/reset is not resource destruction');
  assert(second.buffers.every(buffer => buffer.destroys === 0));
  assert.equal(first.pipeline.kvCache, null);
  assert.equal(first.pipeline.decodeBuffers.hasBuffers(), false);
  await first.pipeline.unload();
  assert(first.buffers.every(buffer => buffer.destroys === 1));
  await second.pipeline.unload();
  assert(second.buffers.every(buffer => buffer.destroys === 1));
});

test('loader cleanup failure does not skip session or storage cleanup', async () => {
  const { pipeline, buffers } = fixture();
  const failure = new Error('injected loader cleanup failure');
  let storageClosed = false;
  pipeline.ownsDopplerLoader = true;
  pipeline.dopplerLoader = { async unload() { throw failure; } };
  pipeline.storageContext = { async close() { storageClosed = true; } };
  await assert.rejects(pipeline.unload(), error => error === failure || error.cause === failure);
  assert(buffers.every(buffer => buffer.destroys === 1));
  assert(storageClosed);
  assert.equal(pipeline.isLoaded, false);
});

test('unload awaits retired buffers without closing a shared pool', async () => {
  const { pipeline } = fixture();
  let release, notifyEntered, entered = false, finished = false;
  const completion = new Promise(resolve => { release = resolve; });
  const entry = new Promise(resolve => { notifyEntered = resolve; });
  pipeline.getBufferPool = () => ({ async settleDeferredDestruction() { entered = true; notifyEntered(); await completion; } });
  const unloading = pipeline.unload().then(() => { finished = true; });
  await Promise.race([entry, unloading]);
  assert.equal(entered, true);
  assert.equal(finished, false);
  release();
  await unloading;
  assert.equal(finished, true);
});

test('unload preserves retirement failure alongside earlier cleanup failure', async () => {
  const { pipeline } = fixture();
  const original = new Error('loader failed'), retirement = new Error('retirement failed');
  pipeline.ownsDopplerLoader = true;
  pipeline.dopplerLoader = { async unload() { throw original; } };
  pipeline.getBufferPool = () => ({ async settleDeferredDestruction() { throw retirement; } });
  await assert.rejects(pipeline.unload(), error => error instanceof AggregateError
    && error.errors[0] === original && error.errors[1] === retirement);
});

test('partial decode allocation failure releases every acquired buffer and permits retry', () => {
  const allocated = [];
  const failure = new Error('injected decode allocation failure');
  let failing = true;
  setDevice({ features: new Set(), limits: {}, createBindGroup() { return {}; }, createBuffer() {
    if (failing && allocated.length === 2) throw failure;
    const buffer = { destroyed: false, destroy() { this.destroyed = true; } };
    allocated.push(buffer);
    return buffer;
  } }, { platformConfig: null });
  try {
    const manager = new DecodeBufferManager();
    const config = { hiddenSize: 256, intermediateSize: 512, activationDtype: 'f32', enablePingPong: true };
    assert.throws(() => manager.ensureBuffers(config), error => error === failure);
    assert(allocated.every(buffer => buffer.destroyed));
    assert.equal(manager.buffers, null);
    failing = false;
    manager.ensureBuffers(config);
    manager.release();
    assert(allocated.every(buffer => buffer.destroyed));
  } finally { setDevice(null, { platformConfig: null }); }
});
