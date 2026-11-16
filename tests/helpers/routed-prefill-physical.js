import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { bootstrapNodeWebGPU, releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { initDevice, getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { configureDeviceMemoryBudget, inspectDeviceMemory } from '../../src/partitions.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';
import { RoutedExpertCache } from '../../src/loader/experts/routed-expert-cache.js';
import { DEFAULT_EXPERT_CACHE_CONFIG } from '../../src/config/schema/runtime/loading.schema.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

/** Short physical byte/lifetime check, never a model or performance qualification. */
export async function runRoutedPrefillPhysical({ openSource = null } = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'routed-prefill-physical-'));
  const bytes = Uint8Array.from({ length: 96 }, (_, i) => i + 1);
  const manifestBytes = encode({ modelId: 'prefill-physical', hashAlgorithm: 'sha256',
    shards: [{ filename: 'weights.bin', size: bytes.length, hash: sha(bytes).slice(7) }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
    files: [{ path: 'weights.bin', size: bytes.length, pieces: Array.from({ length: 24 }, (_, i) => ({
      offset: i * 4, size: 4, identity: sha(bytes.subarray(i * 4, i * 4 + 4)),
    })) }] });
  const gather = { maxPartBytes: 8, memoryBudgetBytes: 32, concurrency: 2 };
  const plan = { layer: 0, numLayers: 2, numExperts: 4, expertBytes: 24, bankBytes: 96, intermediateSize: 2,
    planes: ['gate', 'up', 'down'].map((role, i) => ({ role, name: role, dtype: 'f32', expertBytes: 8, bytes: 32,
      location: { shardIndex: 0, offset: i * 32, size: 32, shape: [4, 2, 1] } })) };
  const policy = { ...DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed, enabled: true, storageMode: 'compact-decode',
    cacheBytes: 256, compactPrefillBytes: 128, compactPrefillReuse: 'resident-gpu',
    prefillCompactAdmission: 'selected-frequency-first', prefillUploadMode: 'direct-bank',
    readConcurrency: 2, stagingBytes: 96, gpuBudgetBytes: 1024 ** 2, reserveBytes: 1024 };
  const owners = [], errors = [], checks = [];
  let device, provider;
  const open = async () => {
    const source = openSource ? await openSource({ directory, manifestBytes, indexBytes, gather })
      : await createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
        acquire: async piece => bytes.slice(piece.offset, piece.offset + piece.size),
        nodeGather: { ...gather, files: { 'weights.bin': join(directory, 'weights.bin') } } });
    const options = { device, policy, compactLayerCount: 2, otherGpuBytes: () => 0,
      rangeLease: source.storage.rangeLease,
      readRange: (shard, offset, length) => source.storage.loadShardRange(shard, offset, length),
      readRangeInto: typeof source.storage.loadShardRangeInto === 'function'
        ? (shard, offset, destination) => source.storage.loadShardRangeInto(shard, offset, destination) : undefined };
    const owner = { source, options, cache: new RoutedExpertCache(options), async close() {
      try { await this.cache.destroy(); } finally { await this.source.storage.close(); }
    } };
    owners.push(owner); return owner;
  };
  const read = async buffer => {
    const staging = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: 'physical_owned_readback' });
    try {
      const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
      const result = new Uint8Array(staging.getMappedRange()).slice(); staging.unmap(); return result;
    } finally { staging.destroy(); }
  };
  const checkFull = async weights => {
    for (const [i, role] of ['gate', 'up', 'down'].entries()) {
      assert.deepEqual(await read(weights[role].buffer), bytes.slice(i * 32, (i + 1) * 32));
    }
  };
  const checkCompact = async (weights, ids) => {
    const map = new Uint32Array((await read(weights.sourceExpertToSlot)).buffer);
    for (const [i, role] of ['gate', 'up', 'down'].entries()) {
      const actual = await read(weights[role].buffer);
      for (const expert of ids) assert.deepEqual(actual.slice(map[expert] * 8, (map[expert] + 1) * 8), bytes.slice(i * 32 + expert * 8, i * 32 + (expert + 1) * 8));
    }
  };
  try {
    await writeFile(join(directory, 'manifest.json'), manifestBytes);
    await writeFile(join(directory, 'pieces.json'), indexBytes);
    await writeFile(join(directory, 'weights.bin'), bytes);
    installNodeFileFetchShim(); provider = await bootstrapNodeWebGPU(); assert(provider.ok, provider.detail);
    device = await initDevice();
    const adapter = getKernelCapabilities().adapterInfo;
    assert(!/swiftshader|llvmpipe|software/i.test(JSON.stringify(adapter)), 'A software adapter cannot qualify physical execution');
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    await configureDeviceMemoryBudget({ maxBytes: 1024 ** 2 });
    const initial = inspectDeviceMemory();
    const b = await open();
    const bWeights = await b.cache.prepare(plan, [0, 1, 2, 3, 3, 2, 3], undefined, 'prefill');
    await checkFull(bWeights); await device.queue.onSubmittedWorkDone();
    const baseline = inspectDeviceMemory();
    const a = await open();
    await checkFull(await a.cache.prepare(plan, [0, 1, 2, 3, 3, 2, 3], undefined, 'prefill'));
    await checkCompact(await a.cache.prepare(plan, [3, 2], undefined, 'decode'), [3, 2]);
    assert.equal(a.cache.getStats().sourceReadBytes, 96);
    assert.equal(a.cache.getStats().prefillSeedBytes, 48);
    await checkFull(await a.cache.prepare(plan, [0, 1, 2, 3], undefined, 'prefill'));
    assert.equal(a.cache.getStats().sourceReadBytes, 144);
    checks.push('physical prefill, selected GPU seeding, first decode and repeat exact-byte reuse');
    await a.close(); await device.queue.onSubmittedWorkDone();
    assert.equal(inspectDeviceMemory().liveBytes, baseline.liveBytes);
    await checkFull(bWeights);
    checks.push('unloading A returns exactly to retained B baseline and B remains usable');

    const cancelled = await open(), entered = deferred(), release = deferred(), controller = new AbortController();
    let hold = true;
    if (cancelled.options.rangeLease) {
      const originalRead = cancelled.options.rangeLease.read;
      cancelled.options.rangeLease = { ...cancelled.options.rangeLease, async read(...args) {
        const lease = await originalRead(...args);
        if (hold) { hold = false; entered.resolve(); await release.promise; }
        return lease;
      } };
    } else {
      const originalRead = cancelled.options.readRangeInto;
      cancelled.options.readRangeInto = async (...args) => {
        const count = await originalRead(...args);
        if (hold) { hold = false; entered.resolve(); await release.promise; }
        return count;
      };
    }
    await cancelled.cache.destroy(); cancelled.cache = new RoutedExpertCache(cancelled.options);
    let settled = false;
    const rejected = assert.rejects(cancelled.cache.prepare(plan, [0], controller.signal, 'prefill'), /Cancel held lease/).then(() => { settled = true; });
    await entered.promise; controller.abort(new Error('Cancel held lease'));
    await new Promise(resolve => setImmediate(resolve)); assert.equal(settled, false);
    release.resolve(); await rejected;
    await checkFull(await cancelled.cache.prepare(plan, [0, 1, 2, 3], undefined, 'prefill'));
    const acquisition = cancelled.source.getGatherReceipt
      ? await cancelled.source.getGatherReceipt() : cancelled.source.getReceipt();
    assert.equal(acquisition.activeReadBytes, 0);
    await cancelled.close();
    checks.push('cancel before upload holds authenticated source ownership until settlement, then recovers');

    const allocation = await open();
    const create = device.createBuffer.bind(device);
    let fail = true;
    device.createBuffer = descriptor => {
      if (fail && descriptor.label === 'compact_0_up') { fail = false; throw new Error('Partial physical allocation'); }
      return create(descriptor);
    };
    try { await assert.rejects(allocation.cache.prepare(plan, [0, 1], undefined, 'prefill'), /Partial physical allocation/); }
    finally { device.createBuffer = create; }
    await checkFull(await allocation.cache.prepare(plan, [0, 1, 2, 3], undefined, 'prefill'));
    await allocation.close();
    checks.push('partial physical allocation retires incomplete compact buffers and retries');

    const publication = await open(), write = device.queue.writeBuffer.bind(device.queue);
    let failMap = true;
    device.queue.writeBuffer = (buffer, ...args) => {
      if (failMap && buffer.label === 'compact_0_source_map') { failMap = false; throw new Error('Before physical publication'); }
      return write(buffer, ...args);
    };
    try { await assert.rejects(publication.cache.prepare(plan, [0, 1], undefined, 'prefill'), /Before physical publication/); }
    finally { device.queue.writeBuffer = write; }
    await checkFull(await publication.cache.prepare(plan, [0, 1, 2, 3], undefined, 'prefill'));
    await publication.close(); await device.queue.onSubmittedWorkDone();
    assert.equal(inspectDeviceMemory().liveBytes, baseline.liveBytes);
    await checkFull(bWeights);
    checks.push('failure before source-map publication drains real GPU copies, retries and preserves B');
    await b.close(); await device.queue.onSubmittedWorkDone();
    const final = inspectDeviceMemory();
    assert.equal(final.liveBytes, initial.liveBytes); assert.deepEqual(errors, []);
    return { scope: 'tiny physical byte/lifetime integration; no model, token or speed qualification', adapter,
      provider: provider.receipt.selectedProviderId, checks, initial, retainedOwnerBaseline: baseline, final };
  } finally {
    await Promise.allSettled(owners.map(owner => owner.close()));
    if (getDevice()) await getDevice().queue.onSubmittedWorkDone();
    destroyDevice(); await releaseNodeWebGPU(); await rm(directory, { recursive: true });
  }
}
