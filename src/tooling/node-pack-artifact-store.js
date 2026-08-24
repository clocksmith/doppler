import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function resolveInsideBase(baseDir, artifactPath) {
  if (typeof artifactPath !== 'string' || !artifactPath.trim() || path.isAbsolute(artifactPath)) {
    throw new Error('Pack artifact path must be a non-empty relative path.');
  }
  return path.resolve(baseDir, artifactPath);
}

export function createNodePackArtifactStore(packPath) {
  const resolvedPackPath = path.resolve(packPath);
  const baseDir = path.dirname(resolvedPackPath);
  return {
    async hashArtifact(artifact) {
      const filePath = resolveInsideBase(baseDir, artifact.path);
      const hash = createHash('sha256');
      let sizeBytes = 0;
      for await (const chunk of createReadStream(filePath)) {
        hash.update(chunk);
        sizeBytes += chunk.byteLength;
      }
      return { hash: `sha256:${hash.digest('hex')}`, sizeBytes };
    },

    async readArtifact(artifact) {
      return new Uint8Array(await fs.readFile(resolveInsideBase(baseDir, artifact.path)));
    },

    resolveArtifactPath(artifact) {
      return resolveInsideBase(baseDir, artifact.path);
    },

    resolveArtifactUrl(artifact) {
      return pathToFileURL(resolveInsideBase(baseDir, artifact.path)).href;
    },
  };
}
