import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { freezePackV2, validatePackExecutable, verifyPackV2Signature, hashPackV2Envelope } from './pack-v2.js';
import { signPackDigest, validatePackSignature, verifyPackDigest } from './pack-signature.js';

export const PACK_V3_SCHEMA_ID = 'doppler.pack/v3';
export const PACK_V3_SCHEMA_VERSION = 3;
const EXECUTABLE_FIELDS = ['modelId', 'modelIR', 'targetPlans', 'wgslModules', 'artifacts', 'program'];

export function getPackV3SemanticPayload(pack) {
  return {
    schema: PACK_V3_SCHEMA_ID,
    schemaVersion: PACK_V3_SCHEMA_VERSION,
    ...Object.fromEntries(EXECUTABLE_FIELDS.map((key) => [key, pack[key]])),
  };
}

export function hashPackV3(pack) {
  return computeCanonicalSha256(getPackV3SemanticPayload(pack));
}

export function validatePackV3(pack, options = {}) {
  if (!pack || typeof pack !== 'object' || Array.isArray(pack)) return { ok: false, errors: ['Pack v3 must be an object.'] };
  const errors = validatePackExecutable(pack).errors;
  const allowed = ['schema', 'schemaVersion', 'packId', 'semanticRoot', 'signature', ...EXECUTABLE_FIELDS];
  for (const key of Object.keys(pack)) if (!allowed.includes(key)) errors.push(`pack.${key} is not allowed in executable Pack v3.`);
  if (pack.schema !== PACK_V3_SCHEMA_ID || pack.schemaVersion !== PACK_V3_SCHEMA_VERSION) errors.push('Invalid Pack v3 schema.');
  const root = hashPackV3(pack);
  if (pack.semanticRoot !== root) errors.push('Pack v3 semanticRoot mismatch.');
  if (pack.packId !== `${pack.modelId}-pack-v3-${root.slice(7)}`) errors.push('Pack v3 packId must bind its complete semantic root.');
  if (options.requireSignature !== false || pack.signature != null) errors.push(...validatePackSignature(pack.signature, root));
  return { ok: errors.length === 0, errors };
}

export function buildPackV3(executable) {
  const payload = structuredClone(getPackV3SemanticPayload(executable));
  const semanticRoot = hashPackV3(payload);
  const pack = { ...payload, packId: `${payload.modelId}-pack-v3-${semanticRoot.slice(7)}`, semanticRoot, signature: null };
  const validation = validatePackV3(pack, { requireSignature: false });
  if (!validation.ok) throw new Error(`Invalid Pack v3: ${validation.errors.join('; ')}`);
  return pack;
}

export async function signPackV3(pack, signer) {
  const snapshot = freezePackV2(structuredClone(pack));
  const validation = validatePackV3(snapshot, { requireSignature: false });
  if (!validation.ok) throw new Error(`Cannot sign Pack v3: ${validation.errors.join('; ')}`);
  return freezePackV2({ ...snapshot, signature: await signPackDigest(snapshot.semanticRoot, signer) });
}

export async function verifyPackV3Signature(pack, trustedSigners) {
  const validation = validatePackV3(pack);
  if (!validation.ok) throw new Error(`Invalid Pack v3: ${validation.errors.join('; ')}`);
  const key = trustedSigners instanceof Map
    ? trustedSigners.get(pack.signature.authority)
    : trustedSigners?.[pack.signature.authority];
  return verifyPackDigest(pack.signature, pack.semanticRoot, key);
}

export async function migratePackV2(pack, { trustedSigners, signer }) {
  const snapshot = freezePackV2(structuredClone(pack));
  await verifyPackV2Signature(snapshot, trustedSigners);
  const migrated = await signPackV3(buildPackV3(snapshot), signer);
  return freezePackV2({
    pack: migrated,
    release: structuredClone(snapshot.release),
    migratedFrom: {
      schema: snapshot.schema,
      semanticRoot: snapshot.semanticRoot,
      envelopeDigest: hashPackV2Envelope(snapshot),
    },
  });
}
