import { createArtifactStorageContext } from '../../storage/artifact-storage-context.js';

export async function createPackArtifactSource(pack, artifactStore) {
  const manifestArtifact = pack.artifacts.find((artifact) => artifact.artifactId === pack.program.manifestArtifactId);
  const origin = 'https://doppler-pack.invalid/';
  const manifestUrl = new URL(manifestArtifact.path, origin);
  const byUrl = new Map();
  for (const artifact of pack.artifacts) {
    const url = new URL(artifact.path, origin);
    if (url.origin !== new URL(origin).origin || url.search || url.hash) throw new Error('Pack artifacts require local, unambiguous paths.');
    if (byUrl.has(url.href)) throw new Error('Pack artifact paths alias the same file.');
    byUrl.set(url.href, artifact);
  }
  const manifestBytes = await artifactStore.readArtifact(manifestArtifact);
  const manifestText = new TextDecoder('utf-8', { fatal: true }).decode(manifestBytes);
  const manifest = JSON.parse(manifestText);
  if (manifest.modelId !== pack.modelId) throw new Error('Pack manifest model identity mismatch.');
  const read = async (path) => {
    const artifact = byUrl.get(new URL(path, manifestUrl).href);
    if (!artifact) throw new Error(`Pack manifest references an artifact outside its signed closure: ${path}.`);
    return artifactStore.readArtifact(artifact);
  };
  const storageContext = createArtifactStorageContext({
    manifest,
    expectedFormat: 'rdrr',
    verifyHashes: true,
    async readRange(path, offset, length) {
      const bytes = await read(path);
      if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0
        || offset + length > bytes.byteLength) throw new Error('Pack artifact range is out of bounds.');
      return bytes.slice(offset, offset + length).buffer;
    },
    async readText(path) { return new TextDecoder('utf-8', { fatal: true }).decode(await read(path)); },
    async readBinary(path) { return (await read(path)).buffer; },
  });
  return {
    modelId: pack.modelId,
    manifest,
    manifestText,
    manifestHash: manifestArtifact.hash.slice('sha256:'.length),
    storageContext,
  };
}
