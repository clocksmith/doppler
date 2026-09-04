import { computeCanonicalSha256, hashBytesSha256 } from '../formats/canonical-hash.js';
import { freezePackV2, hashPackV2Envelope, validatePackV2, verifyPackV2Signature, verifyPackV2Artifacts } from './pack-v2.js';
import { validatePackV3, verifyPackV3Signature } from './pack-v3.js';
import { verifyPackReleaseEvents } from './pack-release-events.js';

export function validatePack(pack, options = {}) {
  if (pack?.schema === 'doppler.pack/v2') return validatePackV2(pack, options);
  if (pack?.schema === 'doppler.pack/v3') return validatePackV3(pack, options);
  return { ok: false, errors: ['Unsupported Doppler Pack schema.'] };
}

export function getPackIdentity(pack) {
  const validation = validatePack(pack);
  if (!validation.ok) throw new Error(validation.errors.join('; '));
  return freezePackV2({
    schema: pack.schema,
    packId: pack.packId,
    semanticRoot: pack.semanticRoot,
    envelopeDigest: hashPackV2Envelope(pack),
    artifactClosureDigest: computeCanonicalSha256(pack.artifacts),
  });
}

export async function verifyPack(pack, options) {
  const snapshot = freezePackV2(structuredClone(pack));
  const validation = validatePack(snapshot);
  if (!validation.ok) throw new Error(`Invalid Pack: ${validation.errors.join('; ')}`);
  const signatureVerifier = snapshot.schema === 'doppler.pack/v3' ? verifyPackV3Signature : verifyPackV2Signature;
  await signatureVerifier(snapshot, options.trustedSigners);
  const lifecycle = snapshot.schema === 'doppler.pack/v3'
    ? await verifyPackReleaseEvents(options.releaseEvents, {
      pack: snapshot, trustedSigners: options.releaseTrustedSigners, policy: options.releasePolicy,
    })
    : null;
  if (typeof options.artifactStore?.readArtifact !== 'function') throw new Error('Pack verification requires artifactStore.readArtifact().');
  const artifactReceipts = await verifyPackV2Artifacts(snapshot, {
    async hashArtifact(artifact) {
      const payload = await options.artifactStore.readArtifact(artifact);
      if (!(payload instanceof Uint8Array) && !(payload instanceof ArrayBuffer)) throw new Error('Pack artifact source must return bytes.');
      return { hash: hashBytesSha256(payload), sizeBytes: payload.byteLength };
    },
  });
  return freezePackV2({ pack: snapshot, identity: getPackIdentity(snapshot), artifactReceipts, lifecycle });
}
