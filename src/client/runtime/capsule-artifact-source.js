import { createArtifactStorageContext } from '../../storage/artifact-storage-context.js';
import { createShaderSourceScope, bindStorageShaderSourceScope } from '../../gpu/kernels/shader-source-scope.js';

export async function createCapsuleArtifactSource(capsule, artifactStore) {
  const manifestArtifact = capsule.artifacts.find((artifact) => artifact.artifactId === capsule.program.manifestArtifactId);
  const origin = 'https://doppler-capsule.invalid/';
  const manifestUrl = new URL(manifestArtifact.path, origin);
  const byUrl = new Map();
  for (const artifact of capsule.artifacts) {
    const url = new URL(artifact.path, origin);
    if (url.origin !== new URL(origin).origin || url.search || url.hash) throw new Error('Capsule artifacts require local, unambiguous paths.');
    if (byUrl.has(url.href)) throw new Error('Capsule artifact paths alias the same file.');
    byUrl.set(url.href, artifact);
  }
  const manifestBytes = await artifactStore.readArtifact(manifestArtifact);
  const manifestText = new TextDecoder('utf-8', { fatal: true }).decode(manifestBytes);
  const manifest = JSON.parse(manifestText);
  if (manifest.modelId !== capsule.modelId) throw new Error('Capsule manifest model identity mismatch.');
  const resolveArtifact = (path) => {
    const artifact = byUrl.get(new URL(path, manifestUrl).href);
    if (!artifact) throw new Error(`Capsule manifest references an artifact outside its signed closure: ${path}.`);
    return artifact;
  };
  const read = (path) => artifactStore.readArtifact(resolveArtifact(path));
  const storageContext = createArtifactStorageContext({
    manifest,
    expectedFormat: 'rdrr',
    verifyHashes: true,
    async readRange(path, offset, length) {
      if (typeof artifactStore.readArtifactRange === 'function') {
        const bytes = await artifactStore.readArtifactRange(resolveArtifact(path), offset, length);
        if (!(bytes instanceof Uint8Array) || bytes.byteLength !== length) throw new Error('Capsule range source returned an invalid byte range.');
        return bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
          ? bytes.buffer : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      }
      const bytes = await read(path);
      if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0
        || offset + length > bytes.byteLength) throw new Error('Capsule artifact range is out of bounds.');
      return bytes.slice(offset, offset + length).buffer;
    },
    async readText(path) { return new TextDecoder('utf-8', { fatal: true }).decode(await read(path)); },
    async readBinary(path) { return (await read(path)).buffer; },
  });
  const sources = new Map();
  for (const module of capsule.wgslModules ?? []) {
    const artifact = capsule.artifacts.find((entry) => entry.artifactId === module.sourceArtifactId);
    if (!artifact || artifact.role !== 'wgsl-source') throw new Error('Capsule WGSL module has no source artifact.');
    const source = new TextDecoder('utf-8', { fatal: true }).decode(await artifactStore.readArtifact(artifact));
    if (sources.has(module.file) && sources.get(module.file) !== source) {
      throw new Error(`Capsule WGSL filename has conflicting sources: ${module.file}.`);
    }
    sources.set(module.file, source);
  }
  bindStorageShaderSourceScope(storageContext, createShaderSourceScope(sources));
  return {
    modelId: capsule.modelId,
    manifest,
    manifestText,
    manifestHash: manifestArtifact.hash.slice('sha256:'.length),
    storageContext,
  };
}
