import { hashBytesSha256 } from '../../formats/canonical-hash.js';
import { assertCapsuleLoadActive, fetchCapsuleBytes, waitForCapsuleRead } from './capsule-acquisition.js';

/** @param {string} baseUrl @param {import('../../config/capsule-v2.js').CapsuleV2Artifact} artifact */
function resolveUrl(baseUrl, artifact) {
  return new URL(artifact.path, baseUrl).href;
}

/** @param {string} capsuleUrl @returns {import('./fetch-capsule-artifact-store.js').FetchCapsuleArtifactStore} */
export function createFetchCapsuleArtifactStore(capsuleUrl) {
  const baseUrl = new URL('.', capsuleUrl).href;
  /** @type {import('./fetch-capsule-artifact-store.js').FetchCapsuleArtifactStore['readArtifact']} */
  const readArtifact = (artifact, options = {}) => {
    if (!Number.isSafeInteger(artifact?.sizeBytes) || artifact.sizeBytes < 0) throw new Error('Capsule artifact requires an exact byte size.');
    return fetchCapsuleBytes(resolveUrl(baseUrl, artifact), options, { phase: 'artifact', artifactId: artifact.artifactId,
      sizeBytes: artifact.sizeBytes, maxBytes: artifact.sizeBytes });
  };
  return {
    async hashArtifact(artifact, options) {
      const bytes = await readArtifact(artifact, options);
      return { hash: hashBytesSha256(bytes), sizeBytes: bytes.byteLength };
    },
    readArtifact,
    async *streamArtifact(artifact, options) {
      const { signal, maxChunkBytes, onLoadProgress } = options;
      if (!Number.isSafeInteger(maxChunkBytes) || maxChunkBytes < 1) throw new Error('Capsule stream requires maxChunkBytes.');
      if (!Number.isSafeInteger(artifact?.sizeBytes) || artifact.sizeBytes < 0) throw new Error('Capsule artifact requires an exact byte size.');
      assertCapsuleLoadActive(signal);
      const response = await fetch(resolveUrl(baseUrl, artifact), { signal });
      /** @type {ReadableStreamDefaultReader<Uint8Array> | undefined} */
      let reader;
      /** @type {ReadableStreamBYOBReader | undefined} */
      let byteReader;
      let completed = false;
      try {
        if (!response.ok) throw new Error(`Capsule artifact fetch failed (${response.status}).`);
        if (!response.body) throw new Error('Capsule artifact response has no readable body.');
        try { byteReader = response.body.getReader({ mode: 'byob' }); }
        catch (error) {
          if (!(error instanceof TypeError)) throw error;
          reader = response.body.getReader();
        }
        let buffer = byteReader ? new Uint8Array(maxChunkBytes) : null;
        let loadedBytes = 0;
        onLoadProgress?.({ phase: 'artifact', artifactId: artifact.artifactId, loadedBytes, totalBytes: artifact.sizeBytes });
        while (true) {
          const next = byteReader && buffer ? byteReader.read(buffer) : reader?.read();
          if (!next) throw new Error('Capsule artifact stream has no active reader.');
          const { value, done } = await waitForCapsuleRead(next, signal);
          assertCapsuleLoadActive(signal);
          if (done) break;
          if (!(value instanceof Uint8Array) || value.buffer.byteLength > maxChunkBytes) throw new Error('Capsule response chunk exceeds its acquisition limit.');
          loadedBytes += value.byteLength;
          if (loadedBytes > artifact.sizeBytes) throw new Error('Capsule artifact exceeds its byte limit.');
          onLoadProgress?.({ phase: 'artifact', artifactId: artifact.artifactId, loadedBytes, totalBytes: artifact.sizeBytes });
          yield value;
          // BYOB transfers the buffer on read; reuse the returned ownership only
          // after the consumer has copied the previous chunk.
          if (byteReader) {
            if (!(value.buffer instanceof ArrayBuffer)) throw new Error('Capsule byte reader must return owned ArrayBuffer storage.');
            buffer = new Uint8Array(value.buffer);
          }
        }
        if (loadedBytes !== artifact.sizeBytes) throw new Error('Capsule artifact size mismatch.');
        completed = true;
      } finally {
        if (!completed) {
          if (byteReader) await byteReader.cancel().catch(() => {});
          else if (reader) await reader.cancel().catch(() => {});
          else await response.body?.cancel().catch(() => {});
        }
        reader?.releaseLock();
        byteReader?.releaseLock();
      }
    },
    resolveArtifactUrl(artifact) {
      return resolveUrl(baseUrl, artifact);
    },
  };
}
