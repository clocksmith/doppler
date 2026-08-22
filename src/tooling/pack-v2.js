/**
 * Node filesystem helpers for the browser-safe Doppler Pack v2 contract.
 *
 * @module tooling/pack-v2
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { stableSortObject } from '../utils/stable-sort-object.js';
import {
  freezePackV2,
  hashPackV2,
  hashPackV2Envelope,
  validatePackV2,
} from '../config/pack-v2.js';

export {
  PACK_V2_PROGRAM_SCHEMA_ID,
  PACK_V2_SCHEMA_ID,
  PACK_V2_SCHEMA_VERSION,
  PACK_V2_SIGNATURE_ALGORITHM,
  buildPackV2,
  freezePackV2,
  getPackV2SemanticPayload,
  hashPackV2,
  hashPackV2Envelope,
  hashPackV2PublicKey,
  signPackV2,
  validatePackV2,
  verifyPackV2,
  verifyPackV2Artifacts,
  verifyPackV2Signature,
} from '../config/pack-v2.js';

/**
 * @param {string} outputPath
 * @param {object} pack
 * @returns {Promise<{ ok: true, outputPath: string, semanticRoot: string, envelopeHash: string }>}
 */
export async function writePackV2(outputPath, pack) {
  const validation = validatePackV2(pack);
  if (!validation.ok) throw new Error(`Cannot write invalid Doppler Pack v2: ${validation.errors.join('; ')}`);
  const resolved = path.resolve(outputPath);
  await fs.mkdir(path.dirname(resolved), { recursive: true });
  await fs.writeFile(resolved, `${JSON.stringify(stableSortObject(pack), null, 2)}\n`, 'utf8');
  return {
    ok: true,
    outputPath: resolved,
    semanticRoot: hashPackV2(pack),
    envelopeHash: hashPackV2Envelope(pack),
  };
}

/**
 * @param {string} packPath
 * @param {{ requireSignature?: boolean }} [options]
 * @returns {Promise<object>}
 */
export async function loadPackV2(packPath, options = {}) {
  const resolved = path.resolve(packPath);
  const pack = JSON.parse(await fs.readFile(resolved, 'utf8'));
  const validation = validatePackV2(pack, options);
  if (!validation.ok) throw new Error(`Invalid Doppler Pack v2 at ${packPath}: ${validation.errors.join('; ')}`);
  return freezePackV2(pack);
}

/**
 * @param {string} value
 * @returns {Promise<JsonWebKey>}
 */
export async function loadPackSigningKey(value) {
  const parsed = JSON.parse(await fs.readFile(path.resolve(value), 'utf8'));
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`Doppler Pack signing key at "${value}" must be a JWK object.`);
  }
  return parsed;
}
