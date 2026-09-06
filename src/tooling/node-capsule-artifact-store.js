import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function resolveInsideBase(baseDir, artifactPath) {
  if (typeof artifactPath !== 'string' || !artifactPath.trim() || path.isAbsolute(artifactPath)) {
    throw new Error('Capsule artifact path must be a non-empty relative path.');
  }
  return path.resolve(baseDir, artifactPath);
}

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

    resolveArtifactPath(artifact) {
      return resolveInsideBase(baseDir, artifact.path);
    },

    resolveArtifactUrl(artifact) {
      return pathToFileURL(resolveInsideBase(baseDir, artifact.path)).href;
    },
  };
}
