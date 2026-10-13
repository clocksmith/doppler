import { resolveLayerPartition, hashLayerPartitionPlan, ORDERED_LAYER_PARTITION_SCHEMA } from './layer-partition-contract.js';
import { resolveGenerationOptions } from '../../../config/generation-contract.js';
import { computeCanonicalSha256 } from '../../../formats/canonical-hash.js';
import { freezeCapsuleV2 } from '../../../config/capsule-v2.js';
import { canonicalGenerationIdentity } from '../../../config/canonical-generation-policy.js';
import { exactRecord } from '../../../config/canonical-contract-codec.js';

const IDENTITY_FIELDS = ['modelId', 'modelIdentity', 'planId', 'threadId', 'attemptId', 'participantA', 'participantB'];
const LIMIT_FIELDS = ['maxTokens', 'maxPromptTokens', 'maxActivationBytes', 'maxOutputCharacters', 'maxAttempts', 'maxConcurrentAttempts'];

/** Public generation binding used by resident admission and stage callers.
 * @type {import('./resident-partition-contract.js').hashPartitionGeneration} */
export function hashPartitionGeneration(generation) {
  return /** @type {`sha256:${string}`} */ (computeCanonicalSha256(resolveGenerationOptions(generation)));
}

/** @type {import('./resident-partition-contract.js').resolveResidentPartitionAllocation} */
export function resolveResidentPartitionAllocation(manifest, manifestHash, input) {
  if (!input || !input.model || input.model.id !== manifest.modelId || input.model.identity !== manifestHash) {
    throw new Error('Resident partition model identity must match the verified Capsule manifest.');
  }
  const allocation = structuredClone(input);
  if (!Number.isSafeInteger(allocation.index) || allocation.index < 0
    || allocation.index >= allocation.plan?.partitions?.length || !allocation.participantId
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
  resolveResidentCanonicalGeneration(allocation);
  return freezeCapsuleV2(allocation);
}

/** @type {import('./resident-partition-contract.js').resolveResidentCanonicalGeneration} */
export function resolveResidentCanonicalGeneration(allocation) {
  if (allocation.canonicalGeneration === undefined) return null;
  if (allocation.plan.schema === ORDERED_LAYER_PARTITION_SCHEMA) {
    throw new Error('Ordered resident canonical target transitions are not implemented.');
  }
  const value = exactRecord(allocation.canonicalGeneration, ['policy', 'executionIdentity']);
  const resolved = canonicalGenerationIdentity(value.policy, value.executionIdentity, allocation.plan.vocabSize);
  if (resolved.policy.maxGeneratedTokens > allocation.limits.maxTokens
    || resolved.policy.materializedInputCapacity > allocation.generation.maxSeqLen
    || resolved.policy.maxRawOutputUtf16 > allocation.limits.maxOutputCharacters
    || resolved.policy.maxVisibleOutputUtf16 > allocation.limits.maxOutputCharacters) {
    throw new Error('Canonical generation exceeds the resident allocation.');
  }
  return resolved;
}

/** @type {import('./resident-partition-contract.js').assertResidentPartitionIdentity} */
export function assertResidentPartitionIdentity(identity, allocation) {
  if (allocation.plan.schema === ORDERED_LAYER_PARTITION_SCHEMA) {
    const ordered = /** @type {import('./resident-partition-contract.js').ResidentOrderedPartitionIdentity} */ (identity);
    exactRecord(ordered, ['schema', 'modelId', 'modelIdentity', 'planId', 'threadId', 'attemptId',
      'requesterId', 'placementGeneration', 'stages']);
    if (ordered.schema !== 'doppler.resident-partition-attempt/v2'
      || ['modelId', 'modelIdentity', 'planId', 'threadId', 'attemptId', 'requesterId'].some(key =>
        typeof Reflect.get(ordered, key) !== 'string' || !Reflect.get(ordered, key))
      || ordered.modelId !== allocation.model.id || ordered.modelIdentity !== allocation.model.identity
      || ordered.planId !== allocation.planId || !Number.isSafeInteger(ordered.placementGeneration)
      || ordered.placementGeneration < 0 || !Array.isArray(ordered.stages)
      || ordered.stages.length !== allocation.plan.stageCount) throw new Error('Ordered resident attempt identity mismatch.');
    for (let index = 0; index < ordered.stages.length; index++) {
      const stage = exactRecord(ordered.stages[index], ['index', 'participantId', 'transportId', 'residentId', 'executionId']);
      if (stage.index !== index || ['participantId', 'transportId', 'residentId', 'executionId'].some(key =>
        typeof stage[key] !== 'string' || !stage[key])) throw new Error('Ordered resident route stage mismatch.');
    }
    if (ordered.stages[allocation.index].participantId !== allocation.participantId
      || new Set(ordered.stages.map(stage => JSON.stringify([stage.participantId, stage.residentId]))).size !== ordered.stages.length) {
      throw new Error('Ordered resident route assignment mismatch.');
    }
    return computeCanonicalSha256(ordered);
  }
  if ('schema' in identity) throw new Error('Ordered resident identity requires an ordered plan.');
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
