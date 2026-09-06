import { computeCanonicalSha256, hashBytesSha256 } from '../formats/canonical-hash.js';
import { freezeCapsuleV2, hashCapsuleV2Envelope, validateCapsuleV2, verifyCapsuleV2Signature, verifyCapsuleV2Artifacts } from './capsule-v2.js';
import { validateCapsuleV3, verifyCapsuleV3Signature } from './capsule-v3.js';
import { verifyCapsuleReleaseEvents } from './capsule-release-events.js';

export function validateCapsule(capsule, options = {}) {
  if (capsule?.schema === 'doppler.capsule/v2') return validateCapsuleV2(capsule, options);
  if (capsule?.schema === 'doppler.capsule/v3') return validateCapsuleV3(capsule, options);
  return { ok: false, errors: ['Unsupported Doppler Capsule schema.'] };
}

export function getCapsuleIdentity(capsule) {
  const validation = validateCapsule(capsule);
  if (!validation.ok) throw new Error(validation.errors.join('; '));
  return freezeCapsuleV2({
    schema: capsule.schema,
    capsuleId: capsule.capsuleId,
    semanticRoot: capsule.semanticRoot,
    envelopeDigest: hashCapsuleV2Envelope(capsule),
    artifactClosureDigest: computeCanonicalSha256(capsule.artifacts),
  });
}

export async function verifyCapsuleMetadata(capsule, options) {
  const snapshot = freezeCapsuleV2(structuredClone(capsule));
  const validation = validateCapsule(snapshot);
  if (!validation.ok) throw new Error(`Invalid Capsule: ${validation.errors.join('; ')}`);
  if (snapshot.schema !== 'doppler.capsule/v3' && options.releasePolicy?.retainedLocalUse !== undefined) {
    throw new Error('Retained local use policy requires Capsule v3.');
  }
  const signatureVerifier = snapshot.schema === 'doppler.capsule/v3' ? verifyCapsuleV3Signature : verifyCapsuleV2Signature;
  await signatureVerifier(snapshot, options.trustedSigners);
  const lifecycle = snapshot.schema === 'doppler.capsule/v3'
    ? await verifyCapsuleReleaseEvents(options.releaseEvents, {
      capsule: snapshot, trustedSigners: options.releaseTrustedSigners, policy: options.releasePolicy,
    })
    : null;
  return freezeCapsuleV2({ capsule: snapshot, identity: getCapsuleIdentity(snapshot), lifecycle });
}

export async function verifyCapsule(capsule, options) {
  const metadata = await verifyCapsuleMetadata(capsule, options);
  if (typeof options.artifactStore?.readArtifact !== 'function') throw new Error('Capsule verification requires artifactStore.readArtifact().');
  const artifactReceipts = await verifyCapsuleV2Artifacts(metadata.capsule, {
    async hashArtifact(artifact) {
      const payload = await options.artifactStore.readArtifact(artifact);
      if (!(payload instanceof Uint8Array) && !(payload instanceof ArrayBuffer)) throw new Error('Capsule artifact source must return bytes.');
      return { hash: hashBytesSha256(payload), sizeBytes: payload.byteLength };
    },
  });
  return freezeCapsuleV2({ ...metadata, artifactReceipts });
}
