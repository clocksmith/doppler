import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { createSha256Hasher } from '../../formats/sha256.js';
import { assertCapsuleLoadActive, waitForCapsuleRead } from './capsule-acquisition.js';
import { normalizeCapsuleLoadingPolicy } from '../../config/capsule-loading.js';

// Explicit host owners only; no process-wide content lookup or mutable backing port.
const backingOwners = new WeakMap();
const BLOCK_BYTES = 65536;

export function createCapsuleArtifactBacking() {
  const owner = Object.freeze({});
  backingOwners.set(owner, new Map());
  return owner;
}

export function createVerifiedCapsuleArtifactStore(capsule, source, options = {}, backing = createCapsuleArtifactBacking()) {
  if (typeof source?.readArtifact !== 'function') throw new Error('Capsule execution requires artifactStore.readArtifact().');
  const shared = backingOwners.get(backing);
  if (!shared) throw new Error('Capsule backing must be an owned createCapsuleArtifactBacking() handle.');
  const { maxRetainedArtifactBytes } = normalizeCapsuleLoadingPolicy(options);
  const artifacts = new Map(capsule.artifacts.map(artifact => [artifact.artifactId, Object.freeze(structuredClone(artifact))]));
  const verified = new Map();
  const snapshots = new Map();
  const pending = new Map();
  let closed = false;
  const metrics = { sourceBytes: 0, hashedBytes: 0, copiedBytes: 0, retainedBytes: 0, peakRetainedBytes: 0, returnedBytes: 0,
    evictions: 0, sourceReadMs: 0, hashingMs: 0, copyingMs: 0,
    backingBytes: 0, peakBackingBytes: 0, backingFiles: 0, snapshotCopiedBytes: 0,
    sharedBackingBytes: 0, peakSnapshotBlockBytes: 0 };
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
    if (borrowed && !pending.has(declared.hash)) return admit(declared, borrowed, true);
    let task = pending.get(declared.hash);
    if (!task) {
      task = (async () => {
        let started = performance.now();
        const payload = await waitForCapsuleRead(source.readArtifact(declared, options), options.signal);
        metrics.sourceReadMs += performance.now() - started;
        assertActive();
        if (!(payload instanceof Uint8Array) && !(payload instanceof ArrayBuffer)) throw new Error('Capsule artifact source must return bytes.');
        const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
        metrics.sourceBytes += bytes.byteLength;
        if (bytes.byteLength !== declared.sizeBytes) throw new Error(`Capsule artifact hash or size mismatch for "${declared.path}".`);
        const chunks = [];
        const hasher = createSha256Hasher();
        // Only owned blocks are hashed; a digest never authenticates mutable source
        // bytes. Each detached allocation is independent of the artifact's size.
        for (let offset = 0; offset < bytes.length; offset += BLOCK_BYTES) {
          assertActive();
          started = performance.now();
          const chunk = Uint8Array.from(bytes.subarray(offset, offset + BLOCK_BYTES));
          metrics.copyingMs += performance.now() - started;
          metrics.copiedBytes += chunk.length;
          metrics.snapshotCopiedBytes += chunk.length;
          metrics.peakSnapshotBlockBytes = Math.max(metrics.peakSnapshotBlockBytes, chunk.length);
          started = performance.now();
          hasher.update(chunk);
          metrics.hashingMs += performance.now() - started;
          metrics.hashedBytes += chunk.length;
          chunks.push(chunk);
        }
        if (`sha256:${hasher.digestHex()}` !== declared.hash) throw new Error(`Capsule artifact hash or size mismatch for "${declared.path}".`);
        assertActive();
        const snapshot = Object.freeze({ size: bytes.length,
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
        return admit(declared, entry, Boolean(existing));
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
  return {
    readArtifact(artifact) { return readArtifactRange(artifact, 0, resolveArtifact(artifact).sizeBytes); },
    readArtifactRange,
    async hashArtifact(artifact) {
      const declared = resolveArtifact(artifact);
      const snapshot = await verifiedSnapshot(declared);
      assertActive();
      return { hash: declared.hash, sizeBytes: snapshot.size };
    },
    getMetrics() { return Object.freeze({ ...metrics }); },
    close() {
      closed = true; verified.clear(); pending.clear();
      for (const [hash, entry] of snapshots) {
        if (--entry.references === 0 && shared.get(hash) === entry) shared.delete(hash);
      }
      snapshots.clear();
      metrics.retainedBytes = 0; metrics.backingBytes = 0; metrics.backingFiles = 0;
    },
  };
}
