import assert from 'node:assert/strict';
import { RoutedExpertCache } from '../../src/loader/experts/routed-expert-cache.js';
import { planRoutedExpertBank } from '../../src/loader/experts/routed-expert-plan.js';

// CPU fault injection only. These checks do not measure physical GPU retirement.
globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function fixture(direct) {
  const manifest = {
    architecture: { numLayers: 1, hiddenSize: 256 },
    moeConfig: { expertFormat: 'qwen3_5', numExperts: 4, expertIntermediateSize: 256 },
  };
  const locations = new Map();
  const shards = new Map();
  for (const [kind, shape] of [['gate_up', [4, 512, 256]], ['down', [4, 256, 256]]]) {
    const shardIndex = shards.size;
    const size = shape[0] * shape[1] * 144;
    const bytes = new Uint8Array(size);
    for (let expert = 0; expert < 4; expert++) {
      bytes.fill(expert + 1 + shardIndex * 16, expert * size / 4, (expert + 1) * size / 4);
    }
    shards.set(shardIndex, bytes);
    locations.set(`model.language_model.layers.0.mlp.experts.${kind}_proj`, {
      shape, size, dtype: 'Q4_K', layout: 'row', shardIndex, offset: 0,
    });
  }
  const plan = planRoutedExpertBank(manifest, locations, 0);
  const buffers = [];
  const reads = [];
  const destinations = new Set();
  const uploads = [];
  const hooks = {
    allocation() {},
    async read() {},
    upload() {},
    async drain() {},
  };
  const gpu = {
    limits: { maxBufferSize: 16 * 1024 ** 2, maxStorageBufferBindingSize: 16 * 1024 ** 2 },
    createBuffer({ size, label, usage }) {
      hooks.allocation();
      const buffer = {
        size, label, usage, data: new Uint8Array(size), destroyed: false,
        destroy() { this.destroyed = true; },
      };
      buffers.push(buffer);
      return buffer;
    },
    queue: {
      onSubmittedWorkDone: () => hooks.drain(),
      writeBuffer(buffer, offset, data) {
        assert.equal(buffer.destroyed, false, 'uploads never target retired buffers');
        hooks.upload(buffer, data);
        buffer.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
        uploads.push({ buffer, bytes: data.byteLength });
      },
    },
  };
  const policy = {
    enabled: true, storageMode: 'compact-decode', evictionPolicy: 'lru',
    compactLayerBytes: null, compactPrefillBytes: 0, compactPrefillReuse: 'none', prefillCompactAdmission: 'none', sourceReadMode: 'plane-serial', prefillUploadMode: 'expert-copy',
    cacheBytes: 2 * plan.expertBytes + plan.numExperts * 4,
    hostCacheBytes: 0, hostCacheAdmission: 'all', hostCacheAdmissionPhase: 'all',
    stagingBytes: 8192, readConcurrency: 2, routeChunkTokens: 2,
    gpuBudgetBytes: 16 * 1024 ** 2, reserveBytes: 1024 ** 2,
  };
  const options = {
    device: gpu, policy, compactLayerCount: 1, otherGpuBytes: () => 0,
    async readRange(shard, offset, length, signal) {
      reads.push({ shard, offset, length });
      await hooks.read(signal);
      return shards.get(shard).slice(offset, offset + length).buffer;
    },
    readRangeInto: direct ? async (shard, offset, destination, signal) => {
      destinations.add(destination.buffer);
      reads.push({ shard, offset, length: destination.byteLength });
      // A cancelled source is allowed to have partially written its destination.
      destination.fill(0xff);
      await hooks.read(signal);
      destination.set(shards.get(shard).subarray(offset, offset + destination.byteLength));
      return destination.byteLength;
    } : undefined,
  };
  return { cache: new RoutedExpertCache(options), options, plan, buffers, reads, destinations, uploads, hooks, policy, shards };
}

function checkWeights(f, weights, experts) {
  const slots = new Uint32Array(weights.sourceExpertToSlot.data.buffer);
  for (const plane of f.plan.planes) {
    for (const expert of experts) {
      assert(slots[expert] < weights.numExperts);
      const offset = slots[expert] * plane.expertBytes;
      assert.deepEqual(weights[plane.role].buffer.data.slice(offset, offset + plane.expertBytes),
        f.shards.get(plane.location.shardIndex).slice(expert * plane.expertBytes, (expert + 1) * plane.expertBytes));
    }
  }
}

for (const direct of [false, true]) {
  // Cancellation cannot settle until both source readers and queued writes drain.
  const f = fixture(direct);
  const entered = deferred(), releaseRead = deferred(), draining = deferred(), releaseDrain = deferred();
  const controller = new AbortController();
  f.hooks.read = async () => { entered.resolve(); await releaseRead.promise; };
  let settled = false;
  const running = f.cache.prepare(f.plan, [0, 1], controller.signal, 'decode');
  const rejected = assert.rejects(running, /cancel source/).then(() => { settled = true; });
  await entered.promise;
  controller.abort(new Error('cancel source'));
  f.hooks.drain = async () => { draining.resolve(); await releaseDrain.promise; };
  releaseRead.resolve();
  await draining.promise;
  assert.equal(settled, false, 'failure waits for queue completion before recycling slots');
  assert.equal(f.cache.getStats().experts, 0, 'incomplete experts are never published');
  assert(f.buffers.every(buffer => !buffer.destroyed), 'the cache retains its pool while draining');
  releaseDrain.resolve();
  await rejected;
  f.hooks.read = async () => {};
  f.hooks.drain = async () => {};
  checkWeights(f, await f.cache.prepare(f.plan, [0, 1], undefined, 'decode'), [0, 1]);
  assert(f.cache.getStats().peakStagingBytes <= f.policy.stagingBytes);
  await f.cache.destroy();
  assert.equal(f.cache.getStats().residentBytes, 0);
  assert(f.buffers.every(buffer => buffer.destroyed));
}

for (const boundary of ['partial-upload', 'before-publication']) {
  const f = fixture(true), controller = new AbortController();
  const cancelAfterBytes = boundary === 'partial-upload' ? 1 : f.plan.expertBytes;
  let uploaded = 0;
  f.hooks.upload = (buffer, data) => {
    if (buffer.label.endsWith('source_map')) return;
    uploaded += data.byteLength;
    if (uploaded >= cancelAfterBytes) controller.abort(new Error('cancel before publication'));
  };
  await assert.rejects(f.cache.prepare(f.plan, [2], controller.signal, 'decode'), /cancel before publication/);
  assert.equal(f.cache.getStats().experts, 0);
  assert(!f.uploads.some(({ buffer }) => buffer.label.endsWith('source_map')));
  f.hooks.upload = () => {};
  checkWeights(f, await f.cache.prepare(f.plan, [2], undefined, 'decode'), [2]);
  await f.cache.destroy();
  assert(f.buffers.every(buffer => buffer.destroyed));
}

{
  const f = fixture(true);
  let allocations = 0;
  f.hooks.allocation = () => { if (++allocations === 2) throw new Error('partial pool allocation'); };
  await assert.rejects(f.cache.prepare(f.plan, [0], undefined, 'decode'), /partial pool allocation/);
  assert.equal(f.cache.getStats().residentBytes, 0);
  assert(f.buffers.every(buffer => buffer.destroyed));
  f.hooks.allocation = () => {};
  checkWeights(f, await f.cache.prepare(f.plan, [0], undefined, 'decode'), [0]);
  await f.cache.destroy();
}

{
  const f = fixture(true);
  await f.cache.prepare(f.plan, [0], undefined, 'decode');
  f.hooks.upload = buffer => {
    if (buffer.label.endsWith('source_map')) throw new Error('source map upload failed');
  };
  await assert.rejects(f.cache.prepare(f.plan, [1], undefined, 'decode'), /source map upload failed/);
  assert.equal(f.cache.getStats().experts, 1, 'only the previous complete expert remains resident');
  f.hooks.upload = () => {};
  const reads = f.reads.length;
  checkWeights(f, await f.cache.prepare(f.plan, [0], undefined, 'decode'), [0]);
  assert.equal(f.reads.length, reads, 'failure preserves an independent completed slot');
  checkWeights(f, await f.cache.prepare(f.plan, [1], undefined, 'decode'), [1]);
  assert(f.reads.length > reads, 'a failed publication must reload its payload');
  await f.cache.destroy();
}

{
  const f = fixture(true), other = new RoutedExpertCache(f.options);
  await f.cache.prepare(f.plan, [0], undefined, 'decode');
  const owned = [...f.buffers];
  const otherWeights = await other.prepare(f.plan, [1], undefined, 'decode');
  const draining = deferred(), release = deferred();
  f.hooks.drain = async () => { draining.resolve(); await release.promise; };
  let closed = false;
  const closing = f.cache.destroy().then(() => { closed = true; });
  await draining.promise;
  assert.equal(closed, false);
  assert(owned.every(buffer => !buffer.destroyed), 'unload waits for deferred GPU completion');
  release.resolve();
  await closing;
  assert(owned.every(buffer => buffer.destroyed));
  assert.equal(otherWeights.down.buffer.destroyed, false, 'unloading one owner preserves the other');
  f.hooks.drain = async () => {};
  checkWeights(f, await other.prepare(f.plan, [1], undefined, 'decode'), [1]);
  await other.destroy();
  assert(f.buffers.every(buffer => buffer.destroyed));
  assert.equal(other.getStats().residentBytes, 0);
}

{
  const f = fixture(true);
  for (const experts of [[0, 1], [2, 3], [0, 1]]) {
    checkWeights(f, await f.cache.prepare(f.plan, experts, undefined, 'decode'), experts);
  }
  assert.equal(f.cache.getStats().stagingAllocations, f.policy.readConcurrency,
    'sequential preparations reuse each reader\'s bounded scratch');
  assert.equal(f.destinations.size, f.policy.readConcurrency);
  f.policy.stagingBytes /= 2;
  checkWeights(f, await f.cache.prepare(f.plan, [2, 3], undefined, 'decode'), [2, 3]);
  assert.equal(f.cache.getStats().stagingAllocations, 2 * f.policy.readConcurrency,
    'a smaller scratch budget replaces the previous reservations');
  assert(f.cache.getStats().peakStagingBytes <= f.policy.stagingBytes);
  await f.cache.destroy();
}

console.log('Compact expert lifecycle: cancellation, allocation/publication failure, deferred drain, recovery and owner isolation passed (CPU mocks).');
