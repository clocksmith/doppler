import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { freezeCapsuleV2, validateCapsuleExecutable, verifyCapsuleV2Signature, hashCapsuleV2Envelope } from './capsule-v2.js';
import { signCapsuleDigest, validateCapsuleSignature, verifyCapsuleDigest } from './capsule-signature.js';

export const CAPSULE_V3_SCHEMA_ID = 'doppler.capsule/v3';
export const CAPSULE_V3_SCHEMA_VERSION = 3;
const EXECUTABLE_FIELDS = ['modelId', 'modelIR', 'targetPlans', 'wgslModules', 'artifacts', 'program'];

export function getCapsuleV3SemanticPayload(capsule) {
  return {
    schema: CAPSULE_V3_SCHEMA_ID,
    schemaVersion: CAPSULE_V3_SCHEMA_VERSION,
    ...Object.fromEntries(EXECUTABLE_FIELDS.map((key) => [key, capsule[key]])),
  };
}

export function hashCapsuleV3(capsule) {
  return computeCanonicalSha256(getCapsuleV3SemanticPayload(capsule));
}

export function validateCapsuleV3(capsule, options = {}) {
  if (!capsule || typeof capsule !== 'object' || Array.isArray(capsule)) return { ok: false, errors: ['Capsule v3 must be an object.'] };
  const errors = validateCapsuleExecutable(capsule).errors;
  const allowed = ['schema', 'schemaVersion', 'capsuleId', 'semanticRoot', 'signature', ...EXECUTABLE_FIELDS];
  for (const key of Object.keys(capsule)) if (!allowed.includes(key)) errors.push(`capsule.${key} is not allowed in executable Capsule v3.`);
  if (capsule.schema !== CAPSULE_V3_SCHEMA_ID || capsule.schemaVersion !== CAPSULE_V3_SCHEMA_VERSION) errors.push('Invalid Capsule v3 schema.');
  const root = hashCapsuleV3(capsule);
  if (capsule.semanticRoot !== root) errors.push('Capsule v3 semanticRoot mismatch.');
  if (capsule.capsuleId !== `${capsule.modelId}-capsule-v3-${root.slice(7)}`) errors.push('Capsule v3 capsuleId must bind its complete semantic root.');
  if (options.requireSignature !== false || capsule.signature != null) errors.push(...validateCapsuleSignature(capsule.signature, root));
  return { ok: errors.length === 0, errors };
}

export function buildCapsuleV3(executable) {
  const payload = structuredClone(getCapsuleV3SemanticPayload(executable));
  const semanticRoot = hashCapsuleV3(payload);
  const capsule = { ...payload, capsuleId: `${payload.modelId}-capsule-v3-${semanticRoot.slice(7)}`, semanticRoot, signature: null };
  const validation = validateCapsuleV3(capsule, { requireSignature: false });
  if (!validation.ok) throw new Error(`Invalid Capsule v3: ${validation.errors.join('; ')}`);
  return capsule;
}

export async function signCapsuleV3(capsule, signer) {
  const snapshot = freezeCapsuleV2(structuredClone(capsule));
  const validation = validateCapsuleV3(snapshot, { requireSignature: false });
  if (!validation.ok) throw new Error(`Cannot sign Capsule v3: ${validation.errors.join('; ')}`);
  return freezeCapsuleV2({ ...snapshot, signature: await signCapsuleDigest(snapshot.semanticRoot, signer) });
}

export async function verifyCapsuleV3Signature(capsule, trustedSigners) {
  const validation = validateCapsuleV3(capsule);
  if (!validation.ok) throw new Error(`Invalid Capsule v3: ${validation.errors.join('; ')}`);
  const key = trustedSigners instanceof Map
    ? trustedSigners.get(capsule.signature.authority)
    : trustedSigners?.[capsule.signature.authority];
  return verifyCapsuleDigest(capsule.signature, capsule.semanticRoot, key);
}

export async function migrateCapsuleV2(capsule, { trustedSigners, signer }) {
  const snapshot = freezeCapsuleV2(structuredClone(capsule));
  await verifyCapsuleV2Signature(snapshot, trustedSigners);
  const migrated = await signCapsuleV3(buildCapsuleV3(snapshot), signer);
  return freezeCapsuleV2({
    capsule: migrated,
    release: structuredClone(snapshot.release),
    migratedFrom: {
      schema: snapshot.schema,
      semanticRoot: snapshot.semanticRoot,
      envelopeDigest: hashCapsuleV2Envelope(snapshot),
    },
  });
}
