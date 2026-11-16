import { computeCanonicalSha256, hashBytesSha256 } from "../../formats/canonical-hash.js";
import { freezeCapsuleV2, hashCapsuleV2Envelope, validateCapsuleV2, verifyCapsuleV2Signature, verifyCapsuleV2Artifacts } from './capsule-v2.js';
import { validateCapsuleV3, verifyCapsuleV3Signature } from './capsule-v3.js';
import { verifyCapsuleReleaseEvents } from './capsule-release-events.js';

/** @param {unknown} capsule @param {{ requireSignature?: boolean }} [options] */
export function validateCapsule(capsule, options = {}) {
  if (!capsule || typeof capsule !== 'object' || !('schema' in capsule)) {
    return { ok: false, errors: ['Unsupported Doppler Capsule schema.'] };
  }
  if (capsule?.schema === 'doppler.capsule/v2') return validateCapsuleV2(capsule, options);
  if (capsule?.schema === 'doppler.capsule/v3') return validateCapsuleV3(capsule, options);
  return { ok: false, errors: ['Unsupported Doppler Capsule schema.'] };
}

/** @param {import('./capsule.js').DopplerCapsule} capsule */
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

/**
 * @param {import('./capsule.js').DopplerCapsule} capsule
 * @param {import('./capsule.js').CapsuleMetadataVerificationOptions} options
 */
export async function verifyCapsuleMetadata(capsule, options) {
  const snapshot = freezeCapsuleV2(structuredClone(capsule));
  const validation = validateCapsule(snapshot);
  if (!validation.ok) throw new Error(`Invalid Capsule: ${validation.errors.join('; ')}`);
  if (snapshot.schema !== 'doppler.capsule/v3' && options.releasePolicy?.retainedLocalUse !== undefined) {
    throw new Error('Retained local use policy requires Capsule v3.');
  }
  if (snapshot.schema === 'doppler.capsule/v3') await verifyCapsuleV3Signature(snapshot, options.trustedSigners);
  else await verifyCapsuleV2Signature(snapshot, options.trustedSigners);
  let lifecycle = null;
  if (snapshot.schema === 'doppler.capsule/v3') {
    if (!options.releaseEvents?.length) throw new Error('Capsule v3 execution requires its signed release event history.');
    if (!options.releasePolicy) throw new Error('Invalid release policy or unknown policy field.');
    if (!options.releaseTrustedSigners) throw new Error('Capsule v3 execution requires trusted release signers.');
    lifecycle = await verifyCapsuleReleaseEvents(options.releaseEvents, {
      capsule: snapshot, trustedSigners: options.releaseTrustedSigners, policy: options.releasePolicy,
    });
  }
  return freezeCapsuleV2({ capsule: snapshot, identity: getCapsuleIdentity(snapshot), lifecycle });
}

/**
 * @param {import('./capsule.js').DopplerCapsule} capsule
 * @param {import('./capsule.js').CapsuleMetadataVerificationOptions & {
 *   artifactStore: { readArtifact(artifact: import('./capsule-v2.js').CapsuleV2Artifact): Promise<Uint8Array | ArrayBuffer> }
 * }} options
 */
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
