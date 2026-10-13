import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { hashLayerPartitionPlan, validateOrderedLayerPartitionPlan } from './layer-partition-plan.js';

/** @param {unknown} value */
const digest = value => typeof value === 'string' && /^sha256:[a-f0-9]{64}$/.test(value);
/** @param {unknown} condition @param {string} message */
const requireValue = (condition, message) => { if (!condition) throw new Error(`Ordered resident qualification: ${message}.`); };

/** Frozen ordered placement, independent reference and exact model facts; no execution or evidence synthesis.
 * @type {import('./ordered-resident-qualification.js').assertOrderedResidentContract} */
export function assertOrderedResidentContract(value) {
  const contract = /** @type {import('./ordered-resident-qualification.js').OrderedResidentQualificationContract} */ (value);
  requireValue(contract?.schema === 'doppler.resident-qualification-contract/v2', 'versioned contract required');
  requireValue([contract.manifestIdentity, contract.executionGraphHash, contract.partitionPlanHash,
    contract.referenceHash, contract.evaluatorHash].every(digest), 'exact model, graph, plan and reference identities required');
  requireValue(Number.isFinite(contract.tolerance) && contract.tolerance > 0
    && Number.isSafeInteger(contract.comparedSteps) && contract.comparedSteps > 0, 'comparison bounds required');
  const plan = contract.partitionPlan;
  validateOrderedLayerPartitionPlan({ modelId: plan?.modelId, architecture: {
    numLayers: plan?.totalLayers, hiddenSize: plan?.hiddenSize, vocabSize: plan?.vocabSize,
  } }, plan);
  requireValue(contract.partitionPlanHash === hashLayerPartitionPlan(plan), 'plan hash mismatch');
  requireValue(Array.isArray(contract.stages) && contract.stages.length === plan.stageCount,
    'one frozen provider placement per ordered stage required');
  requireValue(contract.stages.every((stage, index) => stage.index === index
    && ['browser-webgpu', 'node-webgpu'].includes(stage.surface) && digest(stage.providerIdentityHash)),
  'ordered stage index, physical surface and provider identity required');
  return contract;
}

/** @type {import('./ordered-resident-qualification.js').assertOrderedResidentRecord} */
export function assertOrderedResidentRecord(record) {
  const contract = assertOrderedResidentContract(record.orderedContract);
  const stage = contract.stages[record.partitionIndex];
  requireValue(Number.isSafeInteger(record.partitionIndex) && stage
    && record.operation === 'residentPartition' && record.status === 'passed'
    && record.partitionPlanHash === contract.partitionPlanHash && record.surface === stage.surface
    && record.providerIdentityHash === stage.providerIdentityHash
    && record.comparedSteps === contract.comparedSteps, 'record differs from frozen plan, stage, provider or comparison');
  return contract;
}

/** Every declared placement must be complete before signing or accepting its TargetPlan.
 * Legacy records retain their independently qualified two-stage meaning.
 * @type {import('./ordered-resident-qualification.js').assertOrderedResidentQualificationSet} */
export function assertOrderedResidentQualificationSet(records) {
  const groups = new Map();
  for (const value of records) {
    if (!value || typeof value !== 'object') continue;
    const record = /** @type {import('./ordered-resident-qualification.js').OrderedResidentQualificationRecord} */ (value);
    if (record.orderedContract === undefined) continue;
    const contract = assertOrderedResidentRecord(record);
    const key = computeCanonicalSha256(contract);
    let group = groups.get(key);
    if (!group) { group = { contract, indices: new Set() }; groups.set(key, group); }
    requireValue(!group.indices.has(record.partitionIndex), 'duplicate stage report');
    group.indices.add(record.partitionIndex);
  }
  for (const { contract, indices } of groups.values()) {
    requireValue(indices.size === contract.partitionPlan.stageCount, 'missing ordered stage report');
  }
}
