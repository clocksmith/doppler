import { hashBytesSha256 } from '../../formats/canonical-hash.js';
import { fetchCapsuleBytes } from './capsule-acquisition.js';

function resolveUrl(baseUrl, artifact) {
  return new URL(artifact.path, baseUrl).href;
}

export function createFetchCapsuleArtifactStore(capsuleUrl) {
  const baseUrl = new URL('.', capsuleUrl).href;
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
    resolveArtifactUrl(artifact) {
      return resolveUrl(baseUrl, artifact);
    },
  };
}
