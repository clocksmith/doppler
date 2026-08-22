/**
 * Doppler Pack v2: Immutable Ahead-of-Time Multi-Target Program Format
 *
 * @module tooling/pack-v2
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { validateModelIR, hashModelIR } from '../config/model-ir.js';
import { validateTargetPlan, hashTargetPlan } from '../config/target-plan.js';
import { sha256BytesHex, sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const PACK_V2_SCHEMA_ID = 'doppler.pack/v2';
export const PACK_V2_SCHEMA_VERSION = 2;

/**
 * Validates a Doppler Pack v2 object.
 *
 * @param {object} pack
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validatePackV2(pack) {
  const errors = [];
  if (!pack || typeof pack !== 'object' || Array.isArray(pack)) {
    return { ok: false, errors: ['Doppler Pack v2 must be a non-null object.'] };
  }
  if (pack.schema !== PACK_V2_SCHEMA_ID) {
    errors.push(`schema must be "${PACK_V2_SCHEMA_ID}", received "${pack.schema}".`);
  }
  if (pack.schemaVersion !== PACK_V2_SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${PACK_V2_SCHEMA_VERSION}, received ${pack.schemaVersion}.`);
  }
  if (typeof pack.packId !== 'string' || !pack.packId.trim()) {
    errors.push('packId must be a non-empty string.');
  }
  if (typeof pack.modelId !== 'string' || !pack.modelId.trim()) {
    errors.push('modelId must be a non-empty string.');
  }

  // Validate ModelIR
  const irValidation = validateModelIR(pack.modelIR);
  if (!irValidation.ok) {
    errors.push(...irValidation.errors.map((e) => `modelIR: ${e}`));
  }

  // Validate TargetPlans
  if (!Array.isArray(pack.targetPlans) || pack.targetPlans.length === 0) {
    errors.push('targetPlans must be a non-empty array.');
  } else {
    for (let index = 0; index < pack.targetPlans.length; index += 1) {
      const plan = pack.targetPlans[index];
      const planValidation = validateTargetPlan(plan);
      if (!planValidation.ok) {
        errors.push(...planValidation.errors.map((e) => `targetPlans[${index}]: ${e}`));
      }
    }
  }

  // Validate WGSL Modules
  if (!Array.isArray(pack.wgslModules) || pack.wgslModules.length === 0) {
    errors.push('wgslModules must be a non-empty array.');
  }

  // Validate Artifacts
  if (!Array.isArray(pack.artifacts) || pack.artifacts.length === 0) {
    errors.push('artifacts must be a non-empty array.');
  }

  return {
    ok: errors.length === 0,
    errors,
  };
}

/**
 * Calculates the canonical cryptographic hash of a Doppler Pack v2.
 *
 * @param {object} pack
 * @returns {`sha256:${string}`}
 */
export function hashPackV2(pack) {
  const validation = validatePackV2(pack);
  if (!validation.ok) {
    throw new Error(`Cannot hash invalid Doppler Pack v2: ${validation.errors.join('; ')}`);
  }
  const canonicalJson = JSON.stringify(stableSortObject(pack));
  return `sha256:${sha256Hex(canonicalJson)}`;
}

/**
 * Builds a Doppler Pack v2 structure.
 *
 * @param {object} params
 * @returns {object} Pack v2
 */
export function buildPackV2(params) {
  const modelId = String(params.modelId || '').trim();
  const packId = params.packId || `${modelId}-${Date.now().toString(16)}`;
  const createdAtUtc = params.createdAtUtc || new Date().toISOString();

  const pack = {
    schema: PACK_V2_SCHEMA_ID,
    schemaVersion: PACK_V2_SCHEMA_VERSION,
    packId,
    modelId,
    createdAtUtc,
    modelIR: params.modelIR,
    targetPlans: Array.isArray(params.targetPlans) ? params.targetPlans : [],
    wgslModules: Array.isArray(params.wgslModules) ? params.wgslModules : [],
    artifacts: Array.isArray(params.artifacts) ? params.artifacts : [],
    signature: params.signature ?? null,
  };

  const validation = validatePackV2(pack);
  if (!validation.ok) {
    throw new Error(`Failed to build valid Doppler Pack v2: ${validation.errors.join('; ')}`);
  }
  return pack;
}

/**
 * Writes a Doppler Pack v2 to disk.
 *
 * @param {string} outputPath
 * @param {object} pack
 * @returns {Promise<{ ok: boolean, outputPath: string, packHash: string }>}
 */
export async function writePackV2(outputPath, pack) {
  const validation = validatePackV2(pack);
  if (!validation.ok) {
    throw new Error(`Cannot write invalid Doppler Pack v2: ${validation.errors.join('; ')}`);
  }
  const resolved = path.resolve(outputPath);
  await fs.mkdir(path.dirname(resolved), { recursive: true });
  const raw = `${JSON.stringify(stableSortObject(pack), null, 2)}\n`;
  await fs.writeFile(resolved, raw, 'utf8');

  return {
    ok: true,
    outputPath: resolved,
    packHash: hashPackV2(pack),
  };
}

/**
 * Loads and validates a Doppler Pack v2 from disk.
 *
 * @param {string} packPath
 * @returns {Promise<object>}
 */
export async function loadPackV2(packPath) {
  const resolved = path.resolve(packPath);
  const raw = await fs.readFile(resolved, 'utf8');
  const pack = JSON.parse(raw);
  const validation = validatePackV2(pack);
  if (!validation.ok) {
    throw new Error(`Invalid Doppler Pack v2 at ${packPath}: ${validation.errors.join('; ')}`);
  }
  return pack;
}
