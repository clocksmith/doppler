import { resolveLayerPartition, hashLayerPartitionPlan } from './layer-partition-contract.js';
import { resolveGenerationOptions } from '../../../config/generation-contract.js';
import { computeCanonicalSha256 } from '../../../formats/canonical-hash.js';
import { freezeCapsuleV2 } from '../../../config/capsule-v2.js';

const IDENTITY_FIELDS = ['modelId', 'modelIdentity', 'planId', 'threadId', 'attemptId', 'participantA', 'participantB'];
const LIMIT_FIELDS = ['maxTokens', 'maxPromptTokens', 'maxActivationBytes', 'maxOutputCharacters', 'maxAttempts', 'maxConcurrentAttempts'];

/** @type {import('./resident-partition-contract.js').resolveResidentPartitionAllocation} */
export function resolveResidentPartitionAllocation(manifest, manifestHash, input) {
  if (!input || !input.model || input.model.id !== manifest.modelId || input.model.identity !== manifestHash) {
    throw new Error('Resident partition model identity must match the verified Capsule manifest.');
  }
  const allocation = structuredClone(input);
  if (![0, 1].includes(allocation.index) || !allocation.participantId
    || hashLayerPartitionPlan(allocation.plan) !== allocation.planId) {
    throw new Error('Resident partition requires an exact plan digest, index and participant identity.');
  }
  resolveLayerPartition(manifest, { plan: allocation.plan, index: allocation.index });
  for (const key of LIMIT_FIELDS) {
    const value = allocation.limits?.[/** @type {keyof import('./resident-partition-contract.js').ResidentPartitionLimits} */ (key)];
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`Resident partition requires positive ${key}.`);
  }
  allocation.generation = resolveGenerationOptions(allocation.generation);
  if (allocation.generation.maxTokens > allocation.limits.maxTokens) throw new Error('Generation exceeds the resident output allocation.');
  return freezeCapsuleV2(allocation);
}

/** @type {import('./resident-partition-contract.js').assertResidentPartitionIdentity} */
export function assertResidentPartitionIdentity(identity, allocation) {
  if (!identity || IDENTITY_FIELDS.some(key => typeof identity[/** @type {keyof typeof identity} */ (key)] !== 'string'
    || !identity[/** @type {keyof typeof identity} */ (key)])
    || identity.modelId !== allocation.model.id || identity.modelIdentity !== allocation.model.identity
    || identity.planId !== allocation.planId || identity.participantA === identity.participantB
    || identity[allocation.index === 0 ? 'participantA' : 'participantB'] !== allocation.participantId) {
    throw new Error('Resident partition attempt identity mismatch.');
  }
  if (identity.requesterId !== undefined || identity.placementGeneration !== undefined) {
    if (typeof identity.requesterId !== 'string' || !identity.requesterId
      || typeof identity.placementGeneration !== 'number'
      || !Number.isSafeInteger(identity.placementGeneration) || identity.placementGeneration < 0) {
      throw new Error('Resident requester placement binding mismatch.');
    }
  }
  return JSON.stringify([...IDENTITY_FIELDS.map(key => identity[/** @type {keyof typeof identity} */ (key)]),
    identity.requesterId ?? null, identity.placementGeneration ?? null]);
}
