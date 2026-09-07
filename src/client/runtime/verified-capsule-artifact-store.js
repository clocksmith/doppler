import { computeCanonicalSha256, hashBytesSha256 } from '../../formats/canonical-hash.js';
import { assertCapsuleLoadActive, waitForCapsuleRead } from './capsule-acquisition.js';
import { normalizeCapsuleLoadingPolicy } from '../../config/capsule-loading.js';

export function createVerifiedCapsuleArtifactStore(capsule, source, options = {}) {
  if (typeof source?.readArtifact !== 'function') throw new Error('Capsule execution requires artifactStore.readArtifact().');
  const { maxRetainedArtifactBytes } = normalizeCapsuleLoadingPolicy(options);
  const artifacts = new Map(capsule.artifacts.map((artifact) => [artifact.artifactId, Object.freeze(structuredClone(artifact))]));
  const verified = new Map();
  const pending = new Map();
  let closed = false;
  const metrics = { sourceBytes: 0, hashedBytes: 0, copiedBytes: 0, retainedBytes: 0, peakRetainedBytes: 0, returnedBytes: 0,
    evictions: 0, sourceReadMs: 0, hashingMs: 0, copyingMs: 0 };
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
  async function verifiedBytes(declared) {
    const retained = verified.get(declared.hash);
    if (retained) {
      if (retained.byteLength !== declared.sizeBytes) throw new Error('Capsule artifact size disagrees with shared content.');
      verified.delete(declared.hash); verified.set(declared.hash, retained);
      return retained;
    }
    let task = pending.get(declared.hash);
    if (!task) {
      task = (async () => {
        let started = performance.now();
        const payload = await waitForCapsuleRead(source.readArtifact(declared, options), options.signal);
        metrics.sourceReadMs += performance.now() - started;
        assertActive();
        if (!(payload instanceof Uint8Array) && !(payload instanceof ArrayBuffer)) throw new Error('Capsule artifact source must return bytes.');
        // Buffer.slice() aliases its source; always take an owned Uint8Array copy.
        started = performance.now();
        const bytes = payload instanceof Uint8Array ? Uint8Array.from(payload) : new Uint8Array(payload.slice(0));
        metrics.copyingMs += performance.now() - started;
        metrics.sourceBytes += bytes.byteLength;
        metrics.copiedBytes += bytes.byteLength;
        if (bytes.byteLength !== declared.sizeBytes) throw new Error(`Capsule artifact hash or size mismatch for "${declared.path}".`);
        metrics.hashedBytes += bytes.byteLength;
        started = performance.now();
        const hash = hashBytesSha256(bytes);
        metrics.hashingMs += performance.now() - started;
        if (hash !== declared.hash) {
          throw new Error(`Capsule artifact hash or size mismatch for "${declared.path}".`);
        }
        assertActive();
        if (maxRetainedArtifactBytes === null || bytes.byteLength <= maxRetainedArtifactBytes) {
          while (maxRetainedArtifactBytes !== null && metrics.retainedBytes + bytes.byteLength > maxRetainedArtifactBytes) {
            const [hash, evicted] = verified.entries().next().value;
            verified.delete(hash); metrics.retainedBytes -= evicted.byteLength; metrics.evictions += 1;
          }
          verified.set(declared.hash, bytes);
          metrics.retainedBytes += bytes.byteLength;
          metrics.peakRetainedBytes = Math.max(metrics.peakRetainedBytes, metrics.retainedBytes);
        }
        return bytes;
      })();
      pending.set(declared.hash, task);
      const remove = () => { if (pending.get(declared.hash) === task) pending.delete(declared.hash); };
      task.then(remove, remove);
    }
    const bytes = await task;
    assertActive();
    if (bytes.byteLength !== declared.sizeBytes) throw new Error('Capsule artifact size disagrees with shared content.');
    return bytes;
  }
  async function readArtifactRange(artifact, offset, length) {
    const declared = resolveArtifact(artifact);
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0
      || !Number.isSafeInteger(offset + length) || offset + length > declared.sizeBytes) {
      throw new Error('Capsule artifact range is out of bounds.');
    }
    const bytes = await verifiedBytes(declared);
    assertActive();
    metrics.copiedBytes += length;
    metrics.returnedBytes += length;
    const started = performance.now();
    const result = bytes.slice(offset, offset + length);
    metrics.copyingMs += performance.now() - started;
    return result;
  }
  return {
    readArtifact(artifact) { return readArtifactRange(artifact, 0, resolveArtifact(artifact).sizeBytes); },
    readArtifactRange,
    async hashArtifact(artifact) {
      const declared = resolveArtifact(artifact);
      const bytes = await verifiedBytes(declared);
      assertActive();
      return { hash: declared.hash, sizeBytes: bytes.byteLength };
    },
    getMetrics() { return Object.freeze({ ...metrics }); },
    close() { closed = true; verified.clear(); pending.clear(); metrics.retainedBytes = 0; },
  };
}
