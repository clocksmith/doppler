import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';
import { loadWeights } from '../../src/inference/pipelines/text/loading/init-weights.js';
import { setDevice } from '../../src/gpu/device.js';
import { ShardCache } from '../../src/loader/shard-cache.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
const manifestBytes = encode({ modelId: 'gather-pipeline', hashAlgorithm: 'sha256',
  shards: [{ filename: 'weights.bin', size: 8, hash: sha(bytes).slice(7) }] });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes),
  maxPieceBytes: 4, files: [{ path: 'weights.bin', size: 8, pieces: [0, 4].map(offset =>
    ({ offset, size: 4, identity: sha(bytes.subarray(offset, offset + 4)) })) }] });
const directory = await mkdtemp(join(tmpdir(), 'gather-pipeline-'));
const path = join(directory, 'weights.bin');
let source;
try {
  await writeFile(path, bytes);
  const cancel = new AbortController();
  source = await createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
    acquire: async piece => bytes.slice(piece.offset, piece.offset + piece.size), signal: cancel.signal,
    nodeGather: { files: { 'weights.bin': path }, maxPartBytes: 8, memoryBudgetBytes: 16 } });
  assert(source.storage.rangeLease, 'Storage factory constructs the owned reader');
  setDevice({ features: new Set(), limits: {}, createBindGroup() {}, lost: new Promise(() => {}) }, { platformConfig: null });
  const cache = new ShardCache({ maxEntries: 1 });
  let handoff;
  const stop = new Error('Stop before any model or GPU load');
  const loader = { setQ4KConfig() {}, setLoaderDebugConfig() {}, setPerLayerInputSession() {},
    setTensorsJsonUrl() {}, setTensorsJsonLoader() {}, setManifest() {},
    setCustomShardLoader(load, options) {
      handoff = options;
      cache.setCustomLoader(load, options.verify, { rangeLease: options.rangeLease });
    }, async init() { throw stop; } };
  await assert.rejects(loadWeights(source.manifest, { numLayers: 0 }, { loader, storageContext: source.storage, loadingConfig: { shardCache: { verifyHashes: true } } }), error => error === stop);
  assert.equal(handoff.rangeLease, source.storage.rangeLease, 'Production weight init forwards the actual port');
  assert.equal(typeof handoff.loadShardRangeInto, 'function', 'RDRR artifact context preserves the supplied authenticated range-into port');
  const direct = new Uint8Array(6);
  assert.equal(await handoff.loadShardRangeInto(0, 1, direct), 6);
  assert.deepEqual([...direct], [2, 3, 4, 5, 6, 7]);
  await assert.rejects(handoff.loadShardRangeInto(0, 7, direct), /exceeds/);
  await assert.rejects(handoff.loadShardRangeInto(1, 0, direct), /out of bounds/);
  const lease = await cache.rangeLease.read(0, 1, 6, new AbortController().signal);
  assert.deepEqual([...lease.bytes], [2, 3, 4, 5, 6, 7]); lease.release();
  const receipt = await source.getGatherReceipt();
  assert.equal(receipt.manifestIdentity, sha(manifestBytes));
  assert.equal(receipt.requestedBytes, 6);
  assert.equal(receipt.verifiedBytes, 8);
  assert.equal(source.getReceipt().requestedBytes, 6, 'Generic receipt includes only its explicit readInto, never worker reads');
  const fallback = await source.storage.loadShardRange(0, 0, 4);
  assert.deepEqual([...new Uint8Array(fallback)], [1, 2, 3, 4]);
  const pending = cache.rangeLease.read(0, 0, 4, new AbortController().signal);
  cancel.abort(new Error('Model scope cancelled'));
  await assert.rejects(pending, /scope cancelled/);
  await assert.rejects(cache.rangeLease.read(0, 0, 4, new AbortController().signal), /scope cancelled/);
  const pendingReceipt = source.getGatherReceipt();
  const receiptClosed = assert.rejects(pendingReceipt, /closed/);
  await source.storage.close();
  await receiptClosed;
  await assert.rejects(source.getGatherReceipt(), /closed/);
} finally { setDevice(null, { platformConfig: null }); await source?.storage.close(); await rm(directory, { recursive: true }); }
console.log('verified-piece-gather-pipeline: factory, production handoff, authentication evidence, cancellation and close passed');
