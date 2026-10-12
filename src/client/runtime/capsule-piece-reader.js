import { createVerifiedPieceReader } from '../../storage/verified-piece-reader.js';
import { normalizeCapsuleLoadingPolicy } from '../../config/capsule-loading.js';
import { createCapsuleLoadScope, waitForCapsuleRead } from './capsule-acquisition.js';

/** Internal adapter: signed metadata authenticates piece identities, never the host reader.
 * @type {import('./capsule-piece-reader.js').createCapsulePieceReader} */
export async function createCapsulePieceReader(capsule, verifiedStore, source, options) {
  if (typeof source.readArtifactRange !== 'function') throw new Error('Selective resident acquisition requires artifactStore.readArtifactRange().');
  const manifestArtifact = capsule.artifacts.find(row => row.artifactId === capsule.program.manifestArtifactId);
  const indexArtifact = capsule.artifacts.find(row => row.artifactId === capsule.program.pieceIndexArtifactId);
  if (manifestArtifact?.role !== 'manifest') throw new Error('Signed manifest artifact required.');
  const readSourceRange = source.readArtifactRange.bind(source);
  if (indexArtifact?.role !== 'piece-index') throw new Error('Selective resident acquisition requires a signed piece-index artifact.');
  const manifestBytes = await verifiedStore.readArtifact(manifestArtifact);
  const indexBytes = await verifiedStore.readArtifact(indexArtifact);
  /** @type {import('../../formats/rdrr/index.js').RDRRManifest} */
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  if (!['sha256', 'blake3'].includes(manifest.hashAlgorithm)) {
    throw new Error('Signed piece acquisition requires a supported manifest shard hash.');
  }
  /** @type {{ maxPieceBytes: number; files: Array<{path:string;size:number}> }} */
  const index = JSON.parse(new TextDecoder().decode(indexBytes));
  const { maxAcquisitionChunkBytes } = normalizeCapsuleLoadingPolicy(options);
  if (!Number.isSafeInteger(index.maxPieceBytes) || index.maxPieceBytes > maxAcquisitionChunkBytes) {
    throw new Error('Signed piece size exceeds the host acquisition limit.');
  }
  const origin = new URL('https://doppler-capsule.invalid/'), byPath = new Map();
  for (const artifact of capsule.artifacts) {
    const url = new URL(artifact.path, origin);
    if (url.origin !== origin.origin || url.search || url.hash || byPath.has(url.href)) {
      throw new Error('Capsule piece artifacts require unambiguous local paths.');
    }
    byPath.set(url.href, artifact);
  }
  const weights = new Set(capsule.program.weightArtifactIds);
  const shards = new Map(), files = new Map();
  const base = new URL(manifestArtifact.path, origin);
  for (const [position, shard] of manifest.shards.entries()) {
    const artifact = byPath.get(new URL(shard.filename, base).href);
    if (!artifact || artifact.role !== 'weight-shard' || !weights.has(artifact.artifactId)
      || !/^[a-f0-9]{64}$/.test(shard.hash)
      || (manifest.hashAlgorithm === 'sha256' && artifact.hash !== `sha256:${shard.hash}`)
      || artifact.sizeBytes !== shard.size
      || shards.has(artifact.artifactId)) throw new Error('Piece manifest and signed weight artifact disagree.');
    shards.set(artifact.artifactId, position);
    files.set(shard.filename, artifact);
  }
  // Rig verifies the complete shard under its declared algorithm. At runtime,
  // the signed index binds the exact manifest and authenticates each acquired
  // SHA-256 piece. A partial read cannot verify a BLAKE3 whole-shard hash; it
  // must not compare that digest to the Capsule's distinct SHA-256 digest or
  // report whole-shard verification.
  if (weights.size !== shards.size || weights.size !== capsule.program.weightArtifactIds.length) {
    throw new Error('Piece manifest must bind every unique signed weight artifact.');
  }
  if (index.files?.some(file => !byPath.has(new URL(file.path, base).href))) {
    throw new Error('Piece index file is outside the signed Capsule closure.');
  }
  for (const file of index.files ?? []) {
    const artifact = byPath.get(new URL(file.path, base).href);
    if (!artifact || artifact.sizeBytes !== file.size) throw new Error('Piece index and signed artifact size disagree.');
  }
  const lifetime = createCapsuleLoadScope(options);
  let opened;
  try {
    opened = await createVerifiedPieceReader({ manifestBytes: new Uint8Array(manifestBytes).buffer,
      indexBytes: new Uint8Array(indexBytes).buffer, indexIdentity: indexArtifact.hash,
      signal: lifetime.options.signal,
      async acquire(piece) {
        const artifact = files.get(piece.path);
        if (!artifact) throw new Error('Only declared resident weights use piece acquisition.');
        return waitForCapsuleRead(readSourceRange(artifact, piece.offset, piece.size, lifetime.options), lifetime.options.signal);
      } });
  } catch (error) { lifetime.close(); throw error; }
  return Object.freeze({
    hasArtifact: artifact => shards.has(artifact.artifactId),
    async readRange(artifact, offset, length) {
      return new Uint8Array(await opened.readRange(manifest.shards[shards.get(artifact.artifactId)].filename, offset, length));
    },
    getReceipt: () => ({ mode: /** @type {const} */ ('signed-piece-index'), indexArtifactId: indexArtifact.artifactId,
      declaredWeightArtifactIds: [...weights], ...opened.getReceipt() }),
    async close() {
      lifetime.abort(new Error('Resident piece acquisition closed.'));
      lifetime.close();
      await opened.close();
    },
  });
}
