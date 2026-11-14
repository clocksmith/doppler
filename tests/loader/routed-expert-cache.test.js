import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';
import { prepareRoutedExperts } from '../../src/loader/experts/loader-bindings.js';
import { ggufBlockSourceBytes } from '../../src/formats/gguf/block-storage.js';
import assert from 'node:assert/strict';
import { RoutedExpertCache } from '../../src/loader/experts/routed-expert-cache.js';
import { planRoutedExpertBank } from '../../src/loader/experts/routed-expert-plan.js';
import { validateQwen35RoutedConfig } from '../../src/config/schema/loading.schema.js';

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
class FakeBuffer {
  constructor(size, usage) { this.size = size; this.usage = usage; this.data = new Uint8Array(size); this.destroyed = false; }
  destroy() { this.destroyed = true; }
}
function device() {
  const buffers = [];
  const gpu = {
    limits: { maxBufferSize: 16 * 1024 ** 2, maxStorageBufferBindingSize: 16 * 1024 ** 2 },
    createBuffer({ size, usage }) { const buffer = new FakeBuffer(size, usage); buffers.push(buffer); return buffer; },
    createCommandEncoder() {
      const commands = [];
      return {
        copyBufferToBuffer(from, start, to, end, bytes) { commands.push(() => {
          assert(!from.destroyed && !to.destroyed, 'no retired buffer can be used');
          to.data.set(from.data.subarray(start, start + bytes), end);
        }); },
        finish() { return commands; },
      };
    },
    queue: {
      onSubmittedWorkDone: async () => {},
      writeBuffer(buffer, offset, bytes) { assert(!buffer.destroyed); buffer.data.set(bytes, offset); },
      submit(batch) { for (const commands of batch) for (const command of commands) command(); },
    },
  };
  return { gpu, buffers };
}
function fixture(separate = false, intermediate = 256, native = false, direct = false) {
  const manifest = { architecture: { numLayers: 3, hiddenSize: 256 },
    moeConfig: { expertFormat: 'qwen3_5', numExperts: 4, expertIntermediateSize: intermediate } };
  const locations = new Map(), shards = new Map();
  for (let layer = 0; layer < 3; layer++) {
    for (const [kind, shape] of (separate ? [['gate', [4, intermediate, 256]], ['up', [4, intermediate, 256]], ['down', [4, 256, intermediate]]]
      : [['gate_up', [4, 2 * intermediate, 256]], ['down', [4, 256, intermediate]]])) {
      const dtype = native ? (kind === 'down' ? 'Q4_1' : 'Q4_0') : 'Q4_K';
      const bytes = native ? ggufBlockSourceBytes(shape, dtype) : shape[0] * shape[1] * Math.ceil(shape[2] / 256) * 144;
      const shard = shards.size, data = new Uint8Array(bytes);
      for (let expert = 0; expert < 4; expert++) data.fill(1 + layer * 16 + expert * 3 + shard, expert * bytes / 4, (expert + 1) * bytes / 4);
      shards.set(shard, data);
      locations.set(separate ? `blk.${layer}.ffn_${kind}_exps.weight` : `model.language_model.layers.${layer}.mlp.experts.${kind}_proj`,
        { shape, size: bytes, dtype, layout: 'row', shardIndex: shard, offset: 0 });
    }
  }
  const cacheBytes = [...locations.values()].slice(0, separate ? 3 : 2).reduce((sum, plane) => sum + plane.size / 4, 0) * 2;
  const policy = { enabled: true, cacheBytes, hostCacheBytes: 0, hostCacheAdmission: 'all', hostCacheAdmissionPhase: 'all', stagingBytes: 8192, readConcurrency: 2,
    routeChunkTokens: 2, gpuBudgetBytes: 16 * 1024 ** 2, reserveBytes: 1024 ** 2 };
  const { gpu, buffers } = device(), reads = [];
  let readHook = async () => {};
  const cache = new RoutedExpertCache({ device: gpu, policy, otherGpuBytes: () => 0,
    readRangeInto: direct ? async (shard, offset, destination, signal) => {
      reads.push({ shard, offset, bytes: destination.byteLength });
      await readHook(signal);
      destination.set(shards.get(shard).subarray(offset, offset + destination.byteLength));
      return destination.byteLength;
    } : undefined,
    readRange: async (shard, offset, bytes, signal) => {
      reads.push({ shard, offset, bytes });
      await readHook(signal);
      return shards.get(shard).slice(offset, offset + bytes).buffer;
    } });
  return { manifest, locations, shards, policy, gpu, buffers, reads, cache,
    setReadHook(value) { readHook = value; }, plan(layer) { return planRoutedExpertBank(manifest, locations, layer); } };
}

{
  const f = fixture(), plan = f.plan(0);
  f.policy.hostCacheBytes = plan.expertBytes * 6;
  await f.cache.prepare(plan, [2]);
  await f.cache.prepare(f.plan(1), [0]);
  await f.cache.prepare(f.plan(2), [0]);
  const sourceBytes = f.cache.getStats().sourceReadBytes;
  const restored = await f.cache.prepare(plan, [2]);
  assert.equal(f.cache.getStats().sourceReadBytes, sourceBytes, 'GPU eviction can recover the exact assembled host payload without source reads');
  assert.equal(f.cache.getStats().hostCacheHits, 1);
  for (const plane of plan.planes) {
    assert.deepEqual(restored[plane.role].buffer.data.slice(2 * plane.expertBytes, 3 * plane.expertBytes),
      f.shards.get(plane.location.shardIndex).slice(2 * plane.expertBytes, 3 * plane.expertBytes));
  }
  assert.equal(f.cache.getStats().hostReservedBytes, 0);
  await f.cache.destroy();
  assert.equal(f.cache.getStats().hostCacheBytes, 0);
}

{
  const f = fixture(), plan = f.plan(0);
  f.policy.hostCacheBytes = plan.expertBytes * 6;
  f.policy.hostCacheAdmission = 'repeat-read';
  await f.cache.prepare(plan, [2]);
  assert.equal(f.cache.getStats().hostCacheBytes, 0, 'first-use payload is not retained');
  assert.equal(f.cache.getStats().hostAdmissionSkips, 1);
  await f.cache.prepare(f.plan(1), [0]); await f.cache.prepare(f.plan(2), [0]);
  await f.cache.prepare(plan, [2]);
  assert(f.cache.getStats().hostCacheBytes > 0, 'the second source read admits the expert');
  await f.cache.prepare(f.plan(1), [0]); await f.cache.prepare(f.plan(2), [0]);
  const bytes = f.cache.getStats().sourceReadBytes;
  await f.cache.prepare(plan, [2]);
  assert.equal(f.cache.getStats().sourceReadBytes, bytes);
  assert.equal(f.cache.getStats().hostCacheHits, 1);
  await f.cache.destroy();
}

{
  const f = fixture(), plan = f.plan(0);
  f.policy.hostCacheBytes = plan.expertBytes * 2;
  for (const layer of [0, 1, 2, 0]) await f.cache.prepare(f.plan(layer), [0, 1]);
  const stats = f.cache.getStats();
  assert(stats.hostEvictions > 0);
  assert(stats.peakHostOwnedOrReservedBytes <= f.policy.hostCacheBytes);
  assert.equal(stats.hostReservedBytes, 0);
  await f.cache.destroy();
}

{
  const f = fixture(), plan = f.plan(0);
  f.policy.hostCacheBytes = plan.expertBytes * 2;
  f.setReadHook(async () => { throw new Error('source read failed'); });
  await assert.rejects(f.cache.prepare(plan, [0]), /source read failed/);
  assert.equal(f.cache.getStats().hostCacheBytes, 0, 'partial payloads never enter the host cache');
  assert.equal(f.cache.getStats().hostReservedBytes, 0, 'failed source reads release their host reservation');
  await f.cache.destroy();
}

{
  const f = fixture(), plan = f.plan(0);
  const weights = await f.cache.prepare(plan, [3, 1, 3]);
  assert.equal(f.cache.getStats().sourceReadBytes, 2 * plan.expertBytes, 'only two selected Q4K payloads, including block metadata');
  assert.equal(f.cache.getStats().uploadBytes, 2 * plan.expertBytes);
  assert(f.cache.getStats().peakStagingBytes <= f.policy.stagingBytes);
  for (const plane of plan.planes) {
    for (const expert of [1, 3]) {
      const begin = expert * plane.expertBytes;
      assert.deepEqual(weights[plane.role].buffer.data.slice(begin, begin + plane.expertBytes),
        f.shards.get(plane.location.shardIndex).slice(begin, begin + plane.expertBytes), 'all packed bytes retain original expert offsets');
    }
    const requested = f.reads.filter(r => r.shard === plane.location.shardIndex);
    assert(requested.every(r => [1, 3].some(id => r.offset >= id * plane.expertBytes && r.offset + r.bytes <= (id + 1) * plane.expertBytes)));
  }
  const reads = f.reads.length;
  await f.cache.prepare(plan, [1, 3]);
  assert.equal(f.reads.length, reads, 'a prepared layer deduplicates across prefill chunks');
  await f.cache.prepare(f.plan(1), [0]);
  assert(f.cache.getStats().evictions > 0);
  assert(f.cache.getStats().cacheBytes <= f.policy.cacheBytes);
  await f.cache.destroy();
  assert(f.buffers.every(b => b.destroyed));
}

{
  const f = fixture();
  f.policy.cacheBytes *= 3;
  await f.cache.prepare(f.plan(0), [2]);
  await f.cache.prepare(f.plan(1), [0]);
  const bytes = f.cache.getStats().sourceReadBytes;
  await f.cache.prepare(f.plan(0), [2]);
  assert.equal(f.cache.getStats().sourceReadBytes, bytes, 'a hot expert survives another layer without another source read');
  assert.equal(f.cache.getStats().cacheHits, 1);
  await f.cache.destroy();
}

{
  const f = fixture(), controller = new AbortController();
  f.policy.hostCacheBytes = f.plan(0).expertBytes * 2;
  let unblock;
  const gate = new Promise(resolve => { unblock = resolve; });
  f.setReadHook(async () => gate);
  const operation = f.cache.prepare(f.plan(0), [0, 1], controller.signal);
  operation.catch(() => {});
  while (!f.reads.length) await new Promise(resolve => setImmediate(resolve));
  assert(f.cache.getStats().hostReservedBytes > 0);
  controller.abort(new DOMException('cancelled', 'AbortError'));
  let settled = false;
  operation.finally(() => { settled = true; }).catch(() => {});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(settled, false, 'cancellation retains ownership until reads settle');
  unblock();
  await assert.rejects(operation, error => error.name === 'AbortError');
  assert.equal(f.cache.getStats().hostCacheBytes, 0, 'cancelled partial payloads cannot be reused');
  assert.equal(f.cache.getStats().hostReservedBytes, 0);
  f.setReadHook(async () => {});
  const retried = await f.cache.prepare(f.plan(0), [0]);
  const plane = f.plan(0).planes[0];
  assert.deepEqual(retried[plane.role].buffer.data.slice(0, plane.expertBytes),
    f.shards.get(plane.location.shardIndex).slice(0, plane.expertBytes));
  await f.cache.destroy();
  assert.equal(f.cache.getStats().hostCacheBytes, 0);
  assert(f.buffers.every(b => b.destroyed));
}

{
  const f = fixture();
  await f.cache.prepare(f.plan(0), [0, 1]);
  const cached = f.buffers.filter(b => b.size === f.plan(0).expertBytes);
  let unblock;
  const fence = new Promise(resolve => { unblock = resolve; });
  f.gpu.queue.onSubmittedWorkDone = () => fence;
  const controller = new AbortController();
  const operation = f.cache.prepare(f.plan(1), [2], controller.signal);
  operation.catch(() => {});
  for (let turn = 0; turn < 50 && f.cache.getStats().evictionFenceWaits === 0; turn++) {
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.equal(f.cache.getStats().evictionFenceWaits, 1, 'the earlier preparation fence cannot cover copies submitted after its snapshot');
  controller.abort(new DOMException('cancelled', 'AbortError'));
  assert(cached.every(b => !b.destroyed), 'a cancelled miss cannot evict buffers before their GPU work completes');
  f.gpu.queue.onSubmittedWorkDone = async () => {};
  unblock();
  await assert.rejects(operation, error => error.name === 'AbortError');
  await f.cache.destroy();
}

{
  const f = fixture();
  f.setReadHook(async () => { throw new Error('source read failed'); });
  await assert.rejects(f.cache.prepare(f.plan(0), [0]), /source read failed/);
  await f.cache.destroy();
  assert(f.buffers.every(b => b.destroyed));
  assert.throws(() => validateQwen35RoutedConfig({ enabled: true }), /cacheBytes/);
  assert.throws(() => f.plan(9), /declared Qwen/);
  f.locations.values().next().value.sourceTransform = { kind: 'unexpected' };
  assert.throws(() => f.plan(0), /untransformed/);
}
console.log('Routed expert cache: range extents, identity, hot reuse, budgets, cancellation, eviction fences and failure cleanup passed.');

{
  const f = fixture(false, 384), plan = f.plan(0);
  const weights = await f.cache.prepare(plan, [2]);
  const down = plan.planes.find(p => p.role === 'down');
  assert.deepEqual(weights.down.shape, [4, 256, 384]);
  assert.equal(down.expertBytes, 256 * 2 * 144, 'Q4K padding belongs to each row, not the whole matrix');
  assert.deepEqual(weights.down.buffer.data.slice(2 * down.expertBytes, 3 * down.expertBytes),
    f.shards.get(down.location.shardIndex).slice(2 * down.expertBytes, 3 * down.expertBytes));
  await f.cache.destroy();
}

{
  const f = fixture(true), plan = f.plan(0);
  const weights = await f.cache.prepare(plan, [2]);
  assert(weights.gate && weights.up && !weights.gateUp);
  assert.equal(f.cache.getStats().sourceReadBytes, plan.expertBytes);
  await f.cache.destroy();
}

{
  const f = fixture(), plan = f.plan(0), plane = plan.planes[0];
  const original = f.shards.get(plane.location.shardIndex), cut = plane.expertBytes + 100;
  f.shards.set(100, original.slice(0, cut));
  f.shards.set(101, original.slice(cut));
  plane.location.spans = [{ shardIndex: 100, offset: 0, size: cut }, { shardIndex: 101, offset: 0, size: original.length - cut }];
  const weights = await f.cache.prepare(plan, [1]);
  assert.deepEqual(weights.gateUp.buffer.data.slice(plane.expertBytes, 2 * plane.expertBytes), original.slice(plane.expertBytes, 2 * plane.expertBytes));
  await f.cache.destroy();
}

{
  const f = fixture();
  f.policy.hostCacheBytes = f.plan(0).expertBytes * 2;
  let unblock;
  const gate = new Promise(resolve => { unblock = resolve; });
  f.setReadHook(async () => gate);
  const running = f.cache.prepare(f.plan(0), [0]);
  running.catch(() => {});
  while (!f.reads.length) await new Promise(resolve => setImmediate(resolve));
  const closing = f.cache.destroy();
  await new Promise(resolve => setImmediate(resolve));
  assert(f.buffers.some(b => !b.destroyed), 'unload retains resources until its cancelled read settles');
  unblock();
  await assert.rejects(running, error => error.name === 'AbortError');
  await closing;
  assert(f.buffers.every(b => b.destroyed));
  assert.equal(f.cache.getStats().hostCacheBytes, 0);
  assert.equal(f.cache.getStats().hostReservedBytes, 0);
}

{
  const f = fixture();
  await f.cache.prepare(f.plan(0), [0]);
  const buffer = f.buffers[0], destroy = buffer.destroy.bind(buffer);
  buffer.destroy = () => { throw new Error('GPU destroy failed'); };
  await assert.rejects(f.cache.destroy(), AggregateError);
  assert(f.cache.getStats().residentBytes > 0, 'failed cleanup retains its owner for retry');
  buffer.destroy = destroy;
  await f.cache.destroy();
  assert.equal(f.cache.getStats().residentBytes, 0);
}

{
  const f = fixture(), plan = f.plan(0);
  await f.cache.prepare(plan, [1]);
  f.policy.gpuBudgetBytes = f.policy.reserveBytes + plan.bankBytes;
  await f.cache.prepare(plan, [1]);
  assert.equal(f.cache.getStats().cacheBytes, 0, 'a working-bank hit still preserves the activation/KV reserve');
  assert.equal(f.cache.getStats().workingHits, 1);
  await f.cache.destroy();
}

{
  const f = fixture(), create = f.gpu.createBuffer.bind(f.gpu);
  let calls = 0;
  f.gpu.createBuffer = options => {
    if (++calls === 2) throw new Error('GPU allocation failed');
    return create(options);
  };
  await assert.rejects(f.cache.prepare(f.plan(0), [0]), /GPU allocation failed/);
  await f.cache.destroy();
  assert(f.buffers.every(b => b.destroyed), 'partial working-bank allocation remains owned and is destroyed');
}

{
  const f = fixture(), failure = new Error('GPU queue lost');
  f.gpu.queue.onSubmittedWorkDone = async () => { throw failure; };
  await assert.rejects(f.cache.prepare(f.plan(0), [0]), error =>
    error === failure || (error instanceof AggregateError && error.errors.includes(failure)));
  f.gpu.queue.onSubmittedWorkDone = async () => {};
  await f.cache.destroy();
  assert(f.buffers.every(b => b.destroyed));
}

{
  const f = fixture(true, 256, true), plan = f.plan(0);
  assert.deepEqual(plan.planes.map(plane => plane.dtype), ['q4_0', 'q4_0', 'q4_1']);
  assert.deepEqual(plan.planes.map(plane => plane.expertBytes), [256 * 256 / 32 * 18, 256 * 256 / 32 * 18, 256 * 256 / 32 * 20]);
  const weights = await f.cache.prepare(plan, [1, 3]);
  assert.equal(f.cache.getStats().sourceReadBytes, 2 * plan.expertBytes);
  for (const plane of plan.planes) {
    assert.equal(weights[plane.role].dtype, plane.dtype);
    for (const id of [1, 3]) assert.deepEqual(
      weights[plane.role].buffer.data.slice(id * plane.expertBytes, (id + 1) * plane.expertBytes),
      f.shards.get(plane.location.shardIndex).slice(id * plane.expertBytes, (id + 1) * plane.expertBytes));
  }
  const before = f.cache.getStats().sourceReadBytes;
  await f.cache.prepare(plan, [1, 3]);
  assert.equal(f.cache.getStats().sourceReadBytes, before);
  assert(f.cache.getStats().peakAccountedGpuBytes <= f.policy.gpuBudgetBytes - f.policy.reserveBytes);
  assert(f.cache.getStats().peakStagingBytes <= f.policy.stagingBytes);
  await f.cache.destroy();
  assert(f.buffers.every(buffer => buffer.destroyed));
  const down = plan.planes.find(plane => plane.role === 'down');
  down.location.size -= 4;
  assert.throws(() => f.plan(0), /extent/);
}

{
  const f = fixture(), plan = f.plan(0);
  await f.cache.prepare(plan, [0, 1]);
  await f.cache.prepare(f.plan(1), [0, 1]);
  const stats = f.cache.getStats();
  assert.equal(stats.evictionFenceWaits, 0, 'the next preparation already fenced both old payload copies');
  assert.equal(stats.evictionFenceSkips, 2, 'completed LRU payloads need no second queue fence');
  assert.equal(stats.evictions, 2);
  assert(stats.peakAccountedGpuBytes <= f.policy.gpuBudgetBytes - f.policy.reserveBytes);
  await f.cache.destroy();
}

{
  const f = fixture();
  await f.cache.prepare(f.plan(0), [0, 1]);
  const allocations = f.buffers.length;
  const oldPayloads = f.buffers.slice(-2);
  await f.cache.prepare(f.plan(1), [0, 1]);
  assert.equal(f.buffers.length, allocations, 'exact-size completed victims replace payloads without allocation');
  assert(oldPayloads.every(buffer => !buffer.destroyed), 'transferred payloads stay owned');
  assert.equal(f.cache.getStats().bufferReuses, 2);
  for (const [index, plane] of f.plan(1).planes.entries()) {
    const working = f.buffers[index];
    const source = f.shards.get(plane.location.shardIndex);
    assert.deepEqual(working.data.slice(0, 2 * plane.expertBytes), source.slice(0, 2 * plane.expertBytes));
  }
  await f.cache.destroy();
  assert(f.buffers.every(buffer => buffer.destroyed), 'reused resources retire on unload');
}

{
  const f = fixture();
  f.policy.cacheBytes *= 2;
  await f.cache.prepare(f.plan(0), [0, 1]);
  await f.cache.prepare(f.plan(1), [0, 1]);
  const before = f.cache.getStats();
  await f.cache.prepare(f.plan(0), [0, 1, 2]);
  const after = f.cache.getStats();
  assert.equal(after.hitBatchSubmissions - before.hitBatchSubmissions, 1);
  assert.equal(after.batchedHitExperts - before.batchedHitExperts, 2);
  assert.equal(after.expertCopySubmissions - before.expertCopySubmissions, 2, 'two hits use one submit; one miss retains its own submit');
  assert.equal(after.cacheHits - before.cacheHits, 2);
  assert.equal(after.cacheMisses - before.cacheMisses, 1);
  for (const [index, plane] of f.plan(0).planes.entries()) {
    assert.deepEqual(f.buffers[index].data.slice(0, 3 * plane.expertBytes),
      f.shards.get(plane.location.shardIndex).slice(0, 3 * plane.expertBytes));
  }
  await f.cache.destroy();
  assert(f.buffers.every(buffer => buffer.destroyed));
}

{
  const f = fixture();
  f.policy.cacheBytes *= 2;
  await f.cache.prepare(f.plan(0), [0, 1]);
  await f.cache.prepare(f.plan(1), [0, 1]);
  const submit = f.gpu.queue.submit;
  f.gpu.queue.submit = () => { throw new Error('hit batch submit failed'); };
  await assert.rejects(f.cache.prepare(f.plan(0), [0, 1]), /hit batch submit failed/);
  f.gpu.queue.submit = submit;
  await f.cache.prepare(f.plan(2), [0, 1, 2, 3]);
  assert(f.cache.getStats().evictions >= 4, 'failed batch releases every pin for subsequent eviction');
  await f.cache.destroy();
  assert(f.buffers.every(buffer => buffer.destroyed));
}

{
  const f = fixture(), plan = f.plan(0);
  f.policy.hostCacheBytes = plan.expertBytes * 2;
  f.policy.hostCacheAdmissionPhase = 'decode';
  assert.throws(() => f.cache.prepare(plan, [0]), /explicit prefill or decode/);
  await f.cache.prepare(plan, [0, 1], undefined, 'decode');
  const retained = f.cache.getStats().hostCacheBytes;
  await f.cache.prepare(f.plan(1), [0, 1], undefined, 'prefill');
  assert.equal(f.cache.getStats().hostCacheBytes, retained);
  assert.equal(f.cache.getStats().hostEvictions, 0, 'prefill misses cannot evict retained decode payloads');
  const source = f.cache.getStats().sourceReadBytes;
  await f.cache.prepare(plan, [0], undefined, 'prefill');
  assert.equal(f.cache.getStats().sourceReadBytes, source, 'one-token prefill can reuse retained decode payload');
  assert.equal(f.cache.getStats().hostPrefillAdmissionSkips, 2);
  assert.equal(f.cache.getStats().hostPrefillAdmissions, 0);
  assert.equal(f.cache.getStats().hostDecodeAdmissions, 2);
  await f.cache.destroy();
}

{
  const f = fixture();
  f.policy.hostCacheBytes = f.plan(0).expertBytes * 2;
  f.policy.hostCacheAdmission = 'repeat-read';
  f.policy.hostCacheAdmissionPhase = 'decode';
  const sweepPrefill = async () => {
    await f.cache.prepare(f.plan(1), [0], undefined, 'prefill');
    await f.cache.prepare(f.plan(2), [0], undefined, 'prefill');
  };
  await f.cache.prepare(f.plan(0), [0], undefined, 'prefill');
  await sweepPrefill();
  await f.cache.prepare(f.plan(0), [0], undefined, 'decode');
  assert.equal(f.cache.getStats().hostCacheBytes, 0, 'one-token prefill must not seed repeat-read admission');
  await sweepPrefill();
  await f.cache.prepare(f.plan(0), [0], undefined, 'decode');
  assert.equal(f.cache.getStats().hostDecodeAdmissions, 1, 'second decode read admits normally');
  await f.cache.destroy();
}

{
  const f = fixture();
  let forwarded;
  const loader = { manifest: f.manifest, tensorLocations: f.locations,
    getRoutedExpertPolicy: () => f.policy,
    routedExpertCache: { prepare: async (...args) => { forwarded = args; return {}; } } };
  await prepareRoutedExperts(loader, 0, [1], undefined, 'prefill');
  assert.equal(forwarded[3], 'prefill', 'loader forwards one-token prefill explicitly rather than guessing decode');
}

{
  const prior = getRuntimeConfig();
  const f = fixture();
  assert.deepEqual(f.cache.getStats().phaseProfiles, {}, 'disabled profiling creates no buckets');
  try {
    setRuntimeConfig({ shared: { debug: { profiler: { enabled: true } } } });
    await f.cache.prepare(f.plan(0), [0, 1], undefined, 'prefill');
    const before = f.cache.getStats();
    await f.cache.prepare(f.plan(1), [0], undefined, 'decode');
    const after = f.cache.getStats();
    assert.equal(after.phaseProfiles.prefill.preparations, 1);
    assert.equal(after.phaseProfiles.decode.preparations, 1);
    assert.equal(after.phaseProfiles.prefill.counters.sourceReadBytes, before.sourceReadBytes);
    assert.equal(after.phaseProfiles.decode.counters.sourceReadBytes, after.sourceReadBytes - before.sourceReadBytes);
    assert(after.phaseProfiles.prefill.prepareWallMs >= 0);
    assert(after.phaseProfiles.prefill.synchronizationWaitMs >= 0);
    assert(after.phaseProfiles.prefill.peakGpuResidentBytes > 0);
    assert(after.phaseProfiles.prefill.peakStagingBytes > 0);
    after.phaseProfiles.decode.counters.sourceReadBytes = -1;
    assert(f.cache.getStats().phaseProfiles.decode.counters.sourceReadBytes >= 0, 'profile snapshots are detached');
    setRuntimeConfig(prior);
    const profiles = f.cache.getStats().phaseProfiles;
    await f.cache.prepare(f.plan(2), [0], undefined, 'decode');
    assert.deepEqual(f.cache.getStats().phaseProfiles, profiles, 'disabled profiling stops attribution');
  } finally { setRuntimeConfig(prior); await f.cache.destroy(); }
}

{
  const old = fixture(true, 256, true), direct = fixture(true, 256, true, true);
  try {
    const a = await old.cache.prepare(old.plan(0), [0, 2], undefined, 'decode');
    const b = await direct.cache.prepare(direct.plan(0), [0, 2], undefined, 'decode');
    for (const role of ['gate', 'up', 'down']) assert.deepEqual(a[role].buffer.data, b[role].buffer.data);
    assert.deepEqual(old.reads, direct.reads, 'readInto preserves source ranges and ordering');
    assert.equal(old.cache.getStats().sourceReadBytes, direct.cache.getStats().sourceReadBytes);
    assert.equal(old.cache.getStats().uploadBytes, direct.cache.getStats().uploadBytes);
    assert.equal(old.cache.getStats().sourceReadIntoBytes, 0);
    assert.equal(direct.cache.getStats().sourceReadIntoBytes, direct.cache.getStats().sourceReadBytes);
  } finally { await old.cache.destroy(); await direct.cache.destroy(); }
}
