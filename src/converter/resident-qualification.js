import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { resolveLayerPartition, hashLayerPartitionPlan, ORDERED_LAYER_PARTITION_SCHEMA } from '../config/layer-partition-plan.js';
import { assertOrderedResidentContract } from '../config/ordered-resident-qualification.js';

/** @param {unknown} value */
const digest = value => typeof value === 'string' && /^sha256:[a-f0-9]{64}$/.test(value);
/** @param {unknown} condition @param {string} message */
const requireValue = (condition, message) => { if (!condition) throw new Error(`Rig resident qualification: ${message}.`); };

/** Admit a trusted producer's retained comparison, not a new numerical evaluator.
 * The producer must derive its comparison from the identified capture/reference;
 * these structural checks cannot establish the truth of a self-authored receipt.
 * @type {import('./resident-qualification.js').assertResidentQualification}
 */
export function assertResidentQualification(report, { manifest, manifestIdentity, executionGraphHash, initialExecutionIdentity, contract }) {
  const ordered = report?.schema === 'doppler.resident-partition-qualification/v2';
  requireValue((ordered || report?.schema === 'doppler.resident-partition-qualification/v1')
    && report.operation === 'residentPartition' && report.status === 'passed', 'passed resident report required');
  requireValue(report.modelId === manifest.modelId && report.manifestIdentity === manifestIdentity
    && report.executionGraphHash === executionGraphHash, 'model, manifest and execution identity mismatch');
  requireValue(initialExecutionIdentity && computeCanonicalSha256(report.initialExecutionIdentity)
    === computeCanonicalSha256(initialExecutionIdentity), 'initial execution identity mismatch');
  requireValue(['browser-webgpu', 'node-webgpu'].includes(report.surface), 'explicit physical surface required');
  requireValue(ordered === (report.partitionPlan?.schema === ORDERED_LAYER_PARTITION_SCHEMA), 'report and partition plan versions differ');
  resolveLayerPartition(manifest, { plan: report.partitionPlan, index: report.partitionIndex });
  requireValue(report.partitionPlanHash === hashLayerPartitionPlan(report.partitionPlan), 'partition plan hash mismatch');
  if (!contract) throw new Error('Rig resident qualification: frozen contract required.');
  const orderedContract = ordered ? assertOrderedResidentContract(contract) : undefined;
  if (orderedContract) {
    const stage = orderedContract.stages[report.partitionIndex];
    requireValue(orderedContract.manifestIdentity === manifestIdentity && orderedContract.executionGraphHash === executionGraphHash
      && computeCanonicalSha256(orderedContract.partitionPlan) === computeCanonicalSha256(report.partitionPlan), 'ordered contract model or plan differs');
    requireValue(stage && stage.surface === report.surface && report.providerIdentity
      && typeof report.providerIdentity === 'object' && !Array.isArray(report.providerIdentity)
      && Object.keys(report.providerIdentity).length > 0
      && stage.providerIdentityHash === computeCanonicalSha256(report.providerIdentity), 'observed provider differs from frozen stage');
  }
  requireValue((ordered || contract.schema === 'doppler.resident-qualification-contract/v1')
    && [contract.referenceHash, contract.evaluatorHash].every(digest)
    && Number.isFinite(contract.tolerance) && contract.tolerance > 0
    && Number.isSafeInteger(contract.comparedSteps) && contract.comparedSteps > 0, 'frozen qualification contract required');
  const comparison = report.comparison;
  requireValue(comparison?.contractHash === computeCanonicalSha256(contract)
    && comparison.referenceHash === contract.referenceHash && comparison.evaluatorHash === contract.evaluatorHash
    && comparison.tolerance === contract.tolerance && comparison.steps?.length === contract.comparedSteps,
    'comparison differs from frozen reference, evaluator, tolerance or step count');
  requireValue(comparison?.complete === true && [comparison.referenceHash, comparison.captureHash, comparison.evaluatorHash].every(digest),
    'complete comparison and independent reference, capture and evaluator identities required');
  requireValue(Number.isFinite(comparison.tolerance) && comparison.tolerance > 0
    && Array.isArray(comparison.steps) && comparison.steps.length > 0, 'comparison bounds and steps required');
  requireValue(comparison.steps.every((step, index) => step.index === index
    && Number.isFinite(step.maxAbsoluteError) && step.maxAbsoluteError >= 0
    && step.maxAbsoluteError <= comparison.tolerance), 'comparison contains failed, missing or invalid steps');
  return { operation: 'residentPartition', surface: report.surface, status: 'passed',
    partitionPlanHash: report.partitionPlanHash, partitionIndex: report.partitionIndex,
    ...(orderedContract ? { orderedContract, providerIdentityHash: computeCanonicalSha256(report.providerIdentity) } : {}),
    comparedSteps: comparison.steps.length, transcriptHash: computeCanonicalSha256(report) };
}
