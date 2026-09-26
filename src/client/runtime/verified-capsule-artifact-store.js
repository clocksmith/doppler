import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { createSha256Hasher } from '../../formats/sha256.js';
import { assertCapsuleLoadActive, createCapsuleLoadScope, waitForCapsuleRead } from './capsule-acquisition.js';
import { normalizeCapsuleLoadingPolicy } from '../../config/capsule-loading.js';

// Explicit host owners only; no process-wide content lookup or mutable backing port.
const backingOwners = new WeakMap();
const verifiedStores = new WeakSet();
const BLOCK_BYTES = 65536;

export function isVerifiedCapsuleArtifactStore(store) {
  return verifiedStores.has(store);
}

export function createCapsuleArtifactBacking() {
  const owner = Object.freeze({});
  backingOwners.set(owner, new Map());
  return owner;
}

export function createVerifiedCapsuleArtifactStore(capsule, source, options = {}, backing = createCapsuleArtifactBacking()) {
  if (typeof source?.readArtifact !== 'function') throw new Error('Capsule execution requires artifactStore.readArtifact().');
  if (source.streamArtifact != null && typeof source.streamArtifact !== 'function') throw new Error('Capsule streamArtifact must be a function.');
  const shared = backingOwners.get(backing);
  if (!shared) throw new Error('Capsule backing must be an owned createCapsuleArtifactBacking() handle.');
  const { maxRetainedArtifactBytes, maxAcquisitionChunkBytes, verificationYieldBytes,
    maxVerifiedBackingBytes, artifactHashBackend } = normalizeCapsuleLoadingPolicy(options);
  const artifacts = new Map(capsule.artifacts.map(artifact => [artifact.artifactId, Object.freeze(structuredClone(artifact))]));
  const verified = new Map();
  const snapshots = new Map();
  const pending = new Map();
  const acquisitions = new Set();
  const readers = new Set();
  let reservedBytes = 0;
  const nodeHost = typeof process !== 'undefined' && Boolean(process.versions?.node);
  const hashBackend = artifactHashBackend === 'host' ? (nodeHost ? 'node-crypto' : 'javascript') : artifactHashBackend;
  if (hashBackend === 'node-crypto' && !nodeHost) throw new Error('Capsule node-crypto hashing requires Node.');
  async function createArtifactHasher() {
    if (hashBackend === 'javascript') return createSha256Hasher();
    const { createHash } = await import('node:crypto');
    const hash = createHash('sha256');
    return { update(bytes) { hash.update(bytes); }, digestHex() { return hash.digest('hex'); } };
  }
  function assertBackingBudget(bytes) {
    if (maxVerifiedBackingBytes !== null && bytes > maxVerifiedBackingBytes - metrics.backingBytes - reservedBytes) {
      throw new Error('Capsule verified backing exceeds maxVerifiedBackingBytes; release unused backing or raise the explicit budget.');
    }
  }
  function trackRead(action) {
    const task = action();
    readers.add(task);
    const done = () => readers.delete(task);
    task.then(done, done);
    return task;
  }
  function clearBacking() {
    verified.clear();
    for (const [hash, entry] of snapshots) {
      if (--entry.references === 0 && shared.get(hash) === entry) shared.delete(hash);
    }
    snapshots.clear();
    metrics.releasedBackingBytes += metrics.backingBytes;
    metrics.retainedBytes = 0; metrics.backingBytes = 0; metrics.backingFiles = 0;
  }
  let closed = false;
  const metrics = { hashBackend, reservedBackingBytes: 0, peakReservedAndBackingBytes: 0, releasedBackingBytes: 0, sourceBytes: 0, hashedBytes: 0, copiedBytes: 0, retainedBytes: 0, peakRetainedBytes: 0, returnedBytes: 0,
    evictions: 0, sourceReadMs: 0, hashingMs: 0, copyingMs: 0,
    backingBytes: 0, peakBackingBytes: 0, backingFiles: 0, snapshotCopiedBytes: 0,
    sharedBackingBytes: 0, peakSnapshotBlockBytes: 0,
    peakSourceChunkBytes: 0, streamedSourceBytes: 0, verificationYields: 0 };
  function assertActive() {
    if (closed) throw new Error('Verified Capsule artifact store is closed.');
    assertCapsuleLoadActive(options.signal);
  }
  function resolveArtifact(artifact) {
    assertActive();
    const declared = artifacts.get(artifact?.artifactId);
    if (!declared || computeCanonicalSha256(declared) !== computeCanonicalSha256(artifact)) throw new Error('Artifact is outside the signed Capsule closure.');
    return declared;
  }
  function admit(declared, entry, reused) {
    assertActive();
    if (entry.snapshot.size !== declared.sizeBytes) throw new Error('Capsule artifact size disagrees with shared content.');
    entry.references++;
    snapshots.set(declared.hash, entry);
    metrics.backingBytes += entry.snapshot.size;
    metrics.backingFiles = snapshots.size;
    metrics.peakBackingBytes = Math.max(metrics.peakBackingBytes, metrics.backingBytes);
    if (reused) metrics.sharedBackingBytes += entry.snapshot.size;
    return entry.snapshot;
  }
  async function verifiedSnapshot(declared) {
    const retained = snapshots.get(declared.hash);
    if (retained) {
      if (retained.snapshot.size !== declared.sizeBytes) throw new Error('Capsule artifact size disagrees with shared content.');
      return retained.snapshot;
    }
    const borrowed = shared.get(declared.hash);
    if (borrowed && !pending.has(declared.hash)) {
      assertBackingBudget(borrowed.snapshot.size);
      const snapshot = admit(declared, borrowed, true);
      metrics.peakReservedAndBackingBytes = Math.max(metrics.peakReservedAndBackingBytes, reservedBytes + metrics.backingBytes);
      return snapshot;
    }
    let task = pending.get(declared.hash);
    if (!task) {
      assertBackingBudget(declared.sizeBytes);
      const acquisition = createCapsuleLoadScope(options);
      reservedBytes += declared.sizeBytes;
      metrics.reservedBackingBytes = reservedBytes;
      metrics.peakReservedAndBackingBytes = Math.max(metrics.peakReservedAndBackingBytes, reservedBytes + metrics.backingBytes);
      task = (async () => {
        acquisitions.add(acquisition);
        const chunks = [];
        let iterator;
        let completed = false;
        let yieldTimer;
        try {
          const streamed = typeof source.streamArtifact === 'function';
          let started = performance.now();
          if (streamed) {
            iterator = source.streamArtifact(declared, { ...acquisition.options, maxChunkBytes: maxAcquisitionChunkBytes })[Symbol.asyncIterator]();
          } else {
            const payload = await waitForCapsuleRead(source.readArtifact(declared, acquisition.options), acquisition.options.signal);
            if (!(payload instanceof Uint8Array) && !(payload instanceof ArrayBuffer)) throw new Error('Capsule artifact source must return bytes.');
            iterator = [payload instanceof Uint8Array ? payload : new Uint8Array(payload)][Symbol.iterator]();
          }
          metrics.sourceReadMs += performance.now() - started;
          const hasher = await createArtifactHasher();
          assertActive();
          let size = 0;
          let sinceYield = 0;
          let hostTask;
          while (true) {
            assertActive();
            started = performance.now();
            const { value: bytes, done } = await waitForCapsuleRead(Promise.resolve(iterator.next()), acquisition.options.signal);
            metrics.sourceReadMs += performance.now() - started;
            assertActive();
            if (done) break;
            if (!(bytes instanceof Uint8Array) || (streamed && bytes.length === 0)) throw new Error('Capsule artifact source must return nonempty byte chunks.');
            if (streamed && bytes.buffer.byteLength > maxAcquisitionChunkBytes) throw new Error('Capsule artifact chunk exceeds its acquisition limit.');
            metrics.peakSourceChunkBytes = Math.max(metrics.peakSourceChunkBytes, bytes.buffer.byteLength);
            metrics.sourceBytes += bytes.byteLength;
            if (streamed) metrics.streamedSourceBytes += bytes.byteLength;
            if (bytes.byteLength > declared.sizeBytes - size) throw new Error(`Capsule artifact hash or size mismatch for "${declared.path}".`);
            // Hash only private storage, never a replaceable source view. Reblock
            // arbitrary transport boundaries into the existing fixed-size backing.
            for (let offset = 0; offset < bytes.length;) {
              assertActive();
              if (!hostTask) hostTask = new Promise(resolve => { yieldTimer = setTimeout(resolve, 0); });
              const local = size % BLOCK_BYTES;
              started = performance.now();
              if (local === 0) chunks.push(new Uint8Array(Math.min(BLOCK_BYTES, declared.sizeBytes - size)));
              const chunk = chunks[chunks.length - 1];
              const count = Math.min(chunk.length - local, bytes.length - offset);
              chunk.set(bytes.subarray(offset, offset + count), local);
              metrics.copyingMs += performance.now() - started;
              metrics.copiedBytes += count;
              metrics.snapshotCopiedBytes += count;
              metrics.peakSnapshotBlockBytes = Math.max(metrics.peakSnapshotBlockBytes, chunk.length);
              started = performance.now();
              hasher.update(chunk.subarray(local, local + count));
              metrics.hashingMs += performance.now() - started;
              metrics.hashedBytes += count;
              offset += count; size += count; sinceYield += count;
              if (sinceYield >= verificationYieldBytes) {
                await hostTask;
                hostTask = null; sinceYield = 0; metrics.verificationYields++;
                assertActive();
              }
            }
          }
          if (size !== declared.sizeBytes || `sha256:${hasher.digestHex()}` !== declared.hash) throw new Error(`Capsule artifact hash or size mismatch for "${declared.path}".`);
          assertActive();
          const snapshot = Object.freeze({ size,
            readRange(offset, length) {
              const result = new Uint8Array(length);
              let copied = 0;
              while (copied < length) {
                const position = offset + copied;
                const chunk = chunks[Math.floor(position / BLOCK_BYTES)];
                const local = position % BLOCK_BYTES;
                const count = Math.min(chunk.length - local, length - copied);
                result.set(chunk.subarray(local, local + count), copied);
                copied += count;
              }
              return result;
            },
          });
          // Only completed verification is shared. Racing preparations retain their
          // independent source cancellation; a successful publisher wins ownership.
          const existing = shared.get(declared.hash);
          const entry = existing ?? { snapshot, references: 0 };
          if (!existing) shared.set(declared.hash, entry);
          const result = admit(declared, entry, Boolean(existing));
          completed = true;
          return result;
        } finally {
          clearTimeout(yieldTimer);
          if (!completed) {
            acquisition.abort(new DOMException('Artifact acquisition ended.', 'AbortError'));
            chunks.length = 0;
            // A non-cooperative pending next() must not prevent cancellation.
            // Built-in transports observe the signal and release in finally.
            try { Promise.resolve(iterator?.return?.()).catch(() => {}); } catch {}
          }
          reservedBytes -= declared.sizeBytes;
          metrics.reservedBackingBytes = reservedBytes;
          acquisitions.delete(acquisition);
          acquisition.close();
        }
      })();
      pending.set(declared.hash, task);
      const remove = () => { if (pending.get(declared.hash) === task) pending.delete(declared.hash); };
      task.then(remove, remove);
    }
    const snapshot = await task;
    assertActive();
    if (snapshot.size !== declared.sizeBytes) throw new Error('Capsule artifact size disagrees with shared content.');
    return snapshot;
  }
  async function readArtifactRange(artifact, offset, length) {
    const declared = resolveArtifact(artifact);
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0
      || !Number.isSafeInteger(offset + length) || offset + length > declared.sizeBytes) throw new Error('Capsule artifact range is out of bounds.');
    const snapshot = await verifiedSnapshot(declared);
    assertActive();
    const started = performance.now();
    const cached = verified.get(declared.hash);
    if (cached) { verified.delete(declared.hash); verified.set(declared.hash, cached); }
    const result = cached ? cached.slice(offset, offset + length) : snapshot.readRange(offset, length);
    // Weight ranges become model resources; don't pin another full JS copy.
    const cacheable = declared.role !== 'weight-shard' && offset === 0 && length === snapshot.size
      && length > 0 && (maxRetainedArtifactBytes === null || length <= maxRetainedArtifactBytes);
    if (!cached && cacheable) {
      const owned = result.slice();
      while (maxRetainedArtifactBytes !== null && metrics.retainedBytes + length > maxRetainedArtifactBytes) {
        const [hash, evicted] = verified.entries().next().value;
        verified.delete(hash); metrics.retainedBytes -= evicted.byteLength; metrics.evictions++;
      }
      verified.set(declared.hash, owned);
      metrics.retainedBytes += length;
      metrics.peakRetainedBytes = Math.max(metrics.peakRetainedBytes, metrics.retainedBytes);
      metrics.copiedBytes += length;
    }
    metrics.copiedBytes += length;
    metrics.returnedBytes += length;
    metrics.copyingMs += performance.now() - started;
    return result;
  }
  const store = Object.freeze({
    readArtifact(artifact) { return trackRead(() => readArtifactRange(artifact, 0, resolveArtifact(artifact).sizeBytes)); },
    readArtifactRange(artifact, offset, length) { return trackRead(() => readArtifactRange(artifact, offset, length)); },
    hashArtifact(artifact) { return trackRead(async () => {
      const declared = resolveArtifact(artifact);
      const snapshot = await verifiedSnapshot(declared);
      assertActive();
      return { hash: declared.hash, sizeBytes: snapshot.size };
    }); },
    async releaseBacking() {
      while (readers.size) await Promise.allSettled([...readers]);
      clearBacking();
    },
    getMetrics() { return Object.freeze({ ...metrics }); },
    close() {
      closed = true; verified.clear(); pending.clear();
      for (const acquisition of acquisitions) acquisition.abort(new Error('Verified Capsule artifact store is closed.'));
      clearBacking();
    },
  });
  verifiedStores.add(store);
  return store;
}
