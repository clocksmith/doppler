import { computeCanonicalSha256, hashBytesSha256 } from '../../formats/canonical-hash.js';
import { assertCapsuleLoadActive, waitForCapsuleRead } from './capsule-acquisition.js';

export function createVerifiedCapsuleArtifactStore(capsule, source, options = {}) {
  if (typeof source?.readArtifact !== 'function') throw new Error('Capsule execution requires artifactStore.readArtifact().');
  const artifacts = new Map(capsule.artifacts.map((artifact) => [artifact.artifactId, Object.freeze(structuredClone(artifact))]));
  const verified = new Map();
  let closed = false;
  const metrics = { sourceBytes: 0, hashedBytes: 0, copiedBytes: 0, retainedBytes: 0, peakRetainedBytes: 0, returnedBytes: 0 };
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
    let task = verified.get(declared.hash);
    if (!task) {
      task = (async () => {
        const payload = await waitForCapsuleRead(source.readArtifact(declared, options), options.signal);
        assertActive();
        if (!(payload instanceof Uint8Array) && !(payload instanceof ArrayBuffer)) throw new Error('Capsule artifact source must return bytes.');
        // Buffer.slice() aliases its source; always take an owned Uint8Array copy.
        const bytes = payload instanceof Uint8Array ? Uint8Array.from(payload) : new Uint8Array(payload.slice(0));
        metrics.sourceBytes += bytes.byteLength;
        metrics.copiedBytes += bytes.byteLength;
        if (bytes.byteLength !== declared.sizeBytes) throw new Error(`Capsule artifact hash or size mismatch for "${declared.path}".`);
        metrics.hashedBytes += bytes.byteLength;
        if (hashBytesSha256(bytes) !== declared.hash) {
          throw new Error(`Capsule artifact hash or size mismatch for "${declared.path}".`);
        }
        metrics.retainedBytes += bytes.byteLength;
        metrics.peakRetainedBytes = Math.max(metrics.peakRetainedBytes, metrics.retainedBytes);
        return bytes;
      })();
      verified.set(declared.hash, task);
      task.catch(() => { if (verified.get(declared.hash) === task) verified.delete(declared.hash); });
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
    return bytes.slice(offset, offset + length);
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
    close() { closed = true; verified.clear(); metrics.retainedBytes = 0; },
  };
}
