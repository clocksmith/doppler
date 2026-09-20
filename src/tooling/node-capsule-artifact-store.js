import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

/** @param {string} baseDir @param {string} artifactPath */
function resolveInsideBase(baseDir, artifactPath) {
  if (typeof artifactPath !== 'string' || !artifactPath.trim() || path.isAbsolute(artifactPath)) {
    throw new Error('Capsule artifact path must be a non-empty relative path.');
  }
  return path.resolve(baseDir, artifactPath);
}

/** @param {string} capsulePath @returns {import('./node-capsule-artifact-store.js').NodeCapsuleArtifactStore} */
export function createNodeCapsuleArtifactStore(capsulePath) {
  const resolvedCapsulePath = path.resolve(capsulePath);
  const baseDir = path.dirname(resolvedCapsulePath);
  return {
    async hashArtifact(artifact, options = {}) {
      const filePath = resolveInsideBase(baseDir, artifact.path);
      const hash = createHash('sha256');
      let sizeBytes = 0;
      for await (const chunk of createReadStream(filePath, { signal: options.signal ?? undefined })) {
        hash.update(chunk);
        sizeBytes += chunk.byteLength;
      }
      return { hash: `sha256:${hash.digest('hex')}`, sizeBytes };
    },

    async readArtifact(artifact, options = {}) {
      const bytes = await fs.readFile(resolveInsideBase(baseDir, artifact.path), { signal: options.signal ?? undefined });
      options.onLoadProgress?.({ phase: 'artifact', artifactId: artifact.artifactId,
        loadedBytes: bytes.byteLength, totalBytes: artifact.sizeBytes });
      return bytes;
    },

    async *streamArtifact(artifact, { signal, maxChunkBytes, onLoadProgress }) {
      if (!Number.isSafeInteger(maxChunkBytes) || maxChunkBytes < 1) throw new Error('Capsule stream requires maxChunkBytes.');
      if (!Number.isSafeInteger(artifact?.sizeBytes) || artifact.sizeBytes < 0) throw new Error('Capsule artifact requires an exact byte size.');
      signal?.throwIfAborted();
      const file = await fs.open(resolveInsideBase(baseDir, artifact.path), 'r');
      try {
        const buffer = new Uint8Array(maxChunkBytes);
        let loadedBytes = 0;
        while (true) {
          signal?.throwIfAborted();
          const { bytesRead } = await file.read(buffer, 0, buffer.length, loadedBytes);
          signal?.throwIfAborted();
          if (!bytesRead) break;
          loadedBytes += bytesRead;
          if (loadedBytes > artifact.sizeBytes) throw new Error('Capsule artifact exceeds its byte limit.');
          onLoadProgress?.({ phase: 'artifact', artifactId: artifact.artifactId, loadedBytes, totalBytes: artifact.sizeBytes });
          yield buffer.subarray(0, bytesRead);
        }
        if (loadedBytes !== artifact.sizeBytes) throw new Error('Capsule artifact size mismatch.');
      } finally { await file.close(); }
    },

    resolveArtifactPath(artifact) {
      return resolveInsideBase(baseDir, artifact.path);
    },

    resolveArtifactUrl(artifact) {
      return pathToFileURL(resolveInsideBase(baseDir, artifact.path)).href;
    },
  };
}
