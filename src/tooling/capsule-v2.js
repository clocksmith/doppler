
import fs from 'node:fs/promises';
import path from 'node:path';
import { stableSortObject } from '../formats/stable-sort-object.js';
import {
  freezeCapsuleV2,
  hashCapsuleV2,
  hashCapsuleV2Envelope,
  validateCapsuleV2,
} from '../config/capsule-v2.js';

export {
  CAPSULE_V2_PROGRAM_SCHEMA_ID,
  CAPSULE_V2_SCHEMA_ID,
  CAPSULE_V2_SCHEMA_VERSION,
  CAPSULE_V2_SIGNATURE_ALGORITHM,
  buildCapsuleV2,
  freezeCapsuleV2,
  getCapsuleV2SemanticPayload,
  hashCapsuleV2,
  hashCapsuleV2Envelope,
  hashCapsuleV2PublicKey,
  signCapsuleV2,
  validateCapsuleV2,
  verifyCapsuleV2,
  verifyCapsuleV2Artifacts,
  verifyCapsuleV2Signature,
} from '../config/capsule-v2.js';

export async function writeCapsuleV2(outputPath, capsule) {
  const validation = validateCapsuleV2(capsule);
  if (!validation.ok) throw new Error(`Cannot write invalid Doppler Capsule v2: ${validation.errors.join('; ')}`);
  const resolved = path.resolve(outputPath);
  await fs.mkdir(path.dirname(resolved), { recursive: true });
  await fs.writeFile(resolved, `${JSON.stringify(stableSortObject(capsule), null, 2)}\n`, 'utf8');
  return {
    ok: true,
    outputPath: resolved,
    semanticRoot: hashCapsuleV2(capsule),
    envelopeHash: hashCapsuleV2Envelope(capsule),
  };
}

export async function loadCapsuleV2(capsulePath, options = {}) {
  const resolved = path.resolve(capsulePath);
  const capsule = JSON.parse(await fs.readFile(resolved, 'utf8'));
  const validation = validateCapsuleV2(capsule, options);
  if (!validation.ok) throw new Error(`Invalid Doppler Capsule v2 at ${capsulePath}: ${validation.errors.join('; ')}`);
  return freezeCapsuleV2(capsule);
}

export async function loadCapsuleSigningKey(value) {
  const parsed = JSON.parse(await fs.readFile(path.resolve(value), 'utf8'));
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`Doppler Capsule signing key at "${value}" must be a JWK object.`);
  }
  return parsed;
}
