import { computeCanonicalSha256, hashBytesSha256 } from '../../formats/canonical-hash.js';

export function createVerifiedPackArtifactStore(pack, source) {
  if (typeof source?.readArtifact !== 'function') throw new Error('Pack execution requires artifactStore.readArtifact().');
  const artifacts = new Map(pack.artifacts.map((artifact) => [artifact.artifactId, artifact]));
  const verified = new Map();
  let closed = false;
  async function readArtifact(artifact) {
    if (closed) throw new Error('Verified Pack artifact store is closed.');
    const declared = artifacts.get(artifact?.artifactId);
    if (!declared || computeCanonicalSha256(declared) !== computeCanonicalSha256(artifact)) throw new Error('Artifact is outside the signed Pack closure.');
    let task = verified.get(declared.hash);
    if (!task) {
      task = (async () => {
        const payload = await source.readArtifact(declared);
        if (!(payload instanceof Uint8Array) && !(payload instanceof ArrayBuffer)) throw new Error('Pack artifact source must return bytes.');
        // Buffer.slice() aliases its source; always take an owned Uint8Array copy.
        const bytes = payload instanceof Uint8Array ? Uint8Array.from(payload) : new Uint8Array(payload.slice(0));
        if (bytes.byteLength !== declared.sizeBytes || hashBytesSha256(bytes) !== declared.hash) {
          throw new Error(`Pack artifact hash or size mismatch for "${declared.path}".`);
        }
        return bytes;
      })();
      verified.set(declared.hash, task);
      task.catch(() => { if (verified.get(declared.hash) === task) verified.delete(declared.hash); });
    }
    const bytes = await task;
    if (closed) throw new Error('Verified Pack artifact store closed during a read.');
    if (bytes.byteLength !== declared.sizeBytes) throw new Error('Pack artifact size disagrees with shared content.');
    return bytes.slice();
  }
  return {
    readArtifact,
    async hashArtifact(artifact) {
      const bytes = await readArtifact(artifact);
      return { hash: artifact.hash, sizeBytes: bytes.byteLength };
    },
    close() { closed = true; verified.clear(); },
  };
}
