import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';
import { RoutedExpertCache } from '../../src/loader/experts/routed-expert-cache.js';
import { DEFAULT_EXPERT_CACHE_CONFIG, validateQwen35RoutedConfig } from '../../src/config/schema/runtime/loading.schema.js';
import { selectPrefillSeedExperts } from '../../src/loader/experts/routed-expert-prefill.js';

// Real authenticated workers and CPU command mocks. No physical inference claim.
globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const bytes = Uint8Array.from({ length: 96 }, (_, i) => i + 1);
const manifestBytes = encode({ modelId: 'prefill-source', hashAlgorithm: 'sha256',
  shards: [{ filename: 'weights.bin', size: bytes.length, hash: sha(bytes).slice(7) }] });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
  files: [{ path: 'weights.bin', size: bytes.length, pieces: Array.from({ length: 24 }, (_, i) => ({
    offset: i * 4, size: 4, identity: sha(bytes.subarray(i * 4, i * 4 + 4)),
  })) }] });
const directory = await mkdtemp(join(tmpdir(), 'routed-prefill-source-'));
const policy = { ...DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed, enabled: true, storageMode: 'compact-decode',
  cacheBytes: 256, compactPrefillBytes: 128, compactPrefillReuse: 'resident-gpu',
  prefillCompactAdmission: 'selected-frequency-first', prefillUploadMode: 'direct-bank',
  readConcurrency: 2, stagingBytes: 96, gpuBudgetBytes: 4096, reserveBytes: 16 };
const plan = { layer: 0, numLayers: 2, numExperts: 4, expertBytes: 24, bankBytes: 96, intermediateSize: 2,
  planes: ['gate', 'up', 'down'].map((role, i) => ({ role, name: role, dtype: 'f32', expertBytes: 8, bytes: 32,
    location: { shardIndex: 0, offset: i * 32, size: 32, shape: [4, 2, 1] } })) };

function gpuFixture() {
  const buffers = [], copies = [];
  const hooks = { allocation() {}, upload() {}, submit() {}, async drain() {} };
  const device = { limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 },
    createBuffer({ size, usage, label }) {
      hooks.allocation();
      const buffer = { size, usage, label, data: new Uint8Array(size), destroyed: false,
        destroy() { this.destroyed = true; } };
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder() {
      const commands = [];
      return { copyBufferToBuffer(source, start, target, end, count) {
        commands.push(() => { assert(!source.destroyed && !target.destroyed);
          target.data.set(source.data.subarray(start, start + count), end); });
        copies.push({ source, start, target, end, count });
      }, finish() { return commands; } };
    },
    queue: { onSubmittedWorkDone: () => hooks.drain(),
      writeBuffer(target, offset, source) {
        hooks.upload(); assert(!target.destroyed);
        target.data.set(new Uint8Array(source.buffer, source.byteOffset, source.byteLength), offset);
      }, submit(batch) { hooks.submit(); for (const commands of batch) for (const command of commands) command(); } },
  };
  return { device, buffers, copies, hooks };
}
function checkFull(weights, ids) {
  for (const [i, role] of ['gate', 'up', 'down'].entries()) for (const expert of ids) {
    assert.deepEqual(weights[role].buffer.data.slice(expert * 8, (expert + 1) * 8), bytes.slice(i * 32 + expert * 8, i * 32 + (expert + 1) * 8));
  }
}
function checkCompact(weights, ids) {
  const map = new Uint32Array(weights.sourceExpertToSlot.data.buffer);
  for (const [i, role] of ['gate', 'up', 'down'].entries()) for (const expert of ids) {
    assert(map[expert] < weights.numExperts);
    assert.deepEqual(weights[role].buffer.data.slice(map[expert] * 8, (map[expert] + 1) * 8), bytes.slice(i * 32 + expert * 8, i * 32 + (expert + 1) * 8));
  }
}
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}
let source, cache;
try {
  await writeFile(join(directory, 'weights.bin'), bytes);
  source = await createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
    acquire: async piece => bytes.slice(piece.offset, piece.offset + piece.size),
    nodeGather: { files: { 'weights.bin': join(directory, 'weights.bin') }, maxPartBytes: 8, memoryBudgetBytes: 32, concurrency: 2 } });
  const gpu = gpuFixture();
  const options = { ...gpu, policy, compactLayerCount: 2, otherGpuBytes: () => 0,
    rangeLease: source.storage.rangeLease,
    async readRange() { throw new Error('Unexpected generic read'); } };
  cache = new RoutedExpertCache(options);
  const full = await cache.prepare(plan, [0, 1, 2, 3, 3, 2, 3], undefined, 'prefill');
  checkFull(full, [0, 1, 2, 3]);
  const first = cache.getStats();
  assert.equal(first.sourceReadBytes, 96);
  assert.equal(first.expertCopySubmissions, 0, 'No per-expert temporary GPU payload or copy');
  assert.equal(first.prefillSeedSubmissions, 1);
  assert.equal(first.prefillSeedBytes, 48);
  assert.equal(first.workingBytes, 96);
  assert.equal(first.stagingAllocations, 0, 'Owned worker output uploads without main scratch');
  assert(first.peakStagingBytes <= policy.stagingBytes);
  assert(gpu.copies.every(copy => copy.source.label.startsWith('routed_working_')), 'Seed comes from the full authenticated working bank');
  checkCompact(await cache.prepare(plan, [3, 2], undefined, 'decode'), [3, 2]);
  assert.equal(cache.getStats().sourceReadBytes, 96, 'First decode reuses the selected prefill experts');
  const again = await cache.prepare(plan, [0, 1, 2, 3], undefined, 'prefill');
  checkFull(again, [0, 1, 2, 3]);
  assert.equal(cache.getStats().prefillGpuReuseExperts, 2);
  assert.equal(cache.getStats().sourceReadBytes, 144);
  await cache.destroy();
  assert(gpu.buffers.every(buffer => buffer.destroyed));

  // A seed submission failure must drain and allow seeding on the next request.
  const failure = gpuFixture();
  cache = new RoutedExpertCache({ ...options, device: failure.device });
  let fail = true, failed = false, settled = false;
  const drainEntered = deferred(), releaseDrain = deferred();
  failure.hooks.submit = () => { if (fail) { fail = false; failed = true; throw new Error('Seed submit failed'); } };
  failure.hooks.drain = async () => { if (failed) { drainEntered.resolve(); await releaseDrain.promise; } };
  const rejected = assert.rejects(cache.prepare(plan, [0, 1, 2, 3], undefined, 'prefill'), /Seed submit failed/)
    .then(() => { settled = true; });
  await drainEntered.promise;
  assert.equal(settled, false, 'Failed seed waits for deferred queue work before recycling slots');
  assert(failure.buffers.every(buffer => !buffer.destroyed));
  releaseDrain.resolve(); await rejected;
  failed = false;
  checkFull(await cache.prepare(plan, [0, 1, 2, 3], undefined, 'prefill'), [0, 1, 2, 3]);
  checkCompact(await cache.prepare(plan, [2, 3], undefined, 'decode'), [2, 3]);
  assert.equal(cache.getStats().prefillSeedSubmissions, 1, 'Failed first seed retries instead of becoming permanently skipped');
  await cache.destroy();

  const partial = gpuFixture();
  let allocations = 0;
  partial.hooks.allocation = () => { if (++allocations === 5) throw new Error('Partial seed allocation'); };
  cache = new RoutedExpertCache({ ...options, device: partial.device });
  await assert.rejects(cache.prepare(plan, [0, 1], undefined, 'prefill'), /Partial seed allocation/);
  assert(partial.buffers.slice(3).every(buffer => buffer.destroyed), 'Incomplete compact allocation retires its own buffers');
  partial.hooks.allocation = () => {};
  checkFull(await cache.prepare(plan, [0, 1], undefined, 'prefill'), [0, 1]);
  checkCompact(await cache.prepare(plan, [1, 0], undefined, 'decode'), [1, 0]);
  await cache.destroy();

  // Cancellation during a direct bank upload releases the worker lease before retry.
  const interrupted = gpuFixture(), cancelUpload = new AbortController();
  cache = new RoutedExpertCache({ ...options, device: interrupted.device });
  interrupted.hooks.upload = () => cancelUpload.abort(new Error('Cancel direct upload'));
  await assert.rejects(cache.prepare(plan, [0, 1], cancelUpload.signal, 'prefill'), /Cancel direct upload/);
  assert.equal((await source.getGatherReceipt()).activeReadBytes, 0);
  interrupted.hooks.upload = () => {};
  checkFull(await cache.prepare(plan, [0, 1], undefined, 'prefill'), [0, 1]);
  await cache.destroy();

  // The generic source mode restores bounded concurrent plane reads, independently of workers.
  const generic = gpuFixture();
  let active = 0, peak = 0;
  const genericOptions = { device: generic.device, policy: { ...policy, sourceReadMode: 'expert-parallel', stagingBytes: 384 },
    compactLayerCount: 2, otherGpuBytes: () => 0, async readRange() { throw new Error('Expected readInto'); },
    async readRangeInto(shard, offset, destination, signal) {
      assert.equal(shard, 0); active++; peak = Math.max(peak, active);
      try { await new Promise(resolve => setImmediate(resolve)); signal.throwIfAborted();
        destination.set(bytes.subarray(offset, offset + destination.length)); return destination.length;
      } finally { active--; }
    } };
  cache = new RoutedExpertCache(genericOptions);
  checkFull(await cache.prepare(plan, [0, 1, 2, 3], undefined, 'prefill'), [0, 1, 2, 3]);
  assert(peak >= 3 && peak <= 6, 'Planes overlap within two bounded expert payloads');
  checkCompact(await cache.prepare(plan, [3, 2], undefined, 'decode'), [3, 2]);
  assert(cache.getStats().peakStagingBytes <= genericOptions.policy.stagingBytes);
  await cache.destroy();

  const entered = deferred(), releaseReads = deferred(), cancelReads = new AbortController();
  const cancelGpu = gpuFixture();
  cache = new RoutedExpertCache({ ...genericOptions, device: cancelGpu.device,
    async readRangeInto(shard, offset, destination) {
      entered.resolve(); await releaseReads.promise;
      destination.set(bytes.subarray(offset, offset + destination.length)); return destination.length;
    } });
  let cancelled = false;
  const cancellation = assert.rejects(cache.prepare(plan, [0], cancelReads.signal, 'prefill'), /Cancel sibling reads/)
    .then(() => { cancelled = true; });
  await entered.promise; cancelReads.abort(new Error('Cancel sibling reads'));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(cancelled, false, 'Disjoint scratch views stay owned until every sibling read settles');
  releaseReads.resolve(); await cancellation;
  checkFull(await cache.prepare(plan, [0], undefined, 'prefill'), [0]);
  await cache.destroy();

  cache = new RoutedExpertCache({ ...options, policy: genericOptions.policy });
  await assert.rejects(cache.prepare(plan, [0], undefined, 'prefill'), /no lease port/);
  await cache.destroy();

  assert.deepEqual(selectPrefillSeedExperts([0, 1, 1, 0, 2], 2), [0, 1], 'Frequency, then last occurrence, preserves declared expert IDs');
  for (const overrides of [{ prefillCompactAdmission: 'invalid' }, { prefillUploadMode: 'invalid' }, { sourceReadMode: 'invalid' },
    { prefillCompactAdmission: 'selected-frequency-first', compactPrefillBytes: 0 }]) {
    assert.throws(() => validateQwen35RoutedConfig({ ...policy, ...overrides }));
  }
  console.log('routed prefill source: authenticated direct bank, seed/decode/repeat reuse, generic plane overlap, abort, deferred drain/retry and bounds passed (CPU)');
} finally { await cache?.destroy(); await source?.storage.close(); await rm(directory, { recursive: true }); }
