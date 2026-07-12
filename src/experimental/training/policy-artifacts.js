import { validateVerifierGuidedArtifact } from './wgsl-repair.js';

function requireString(value, label) {
  const normalized = typeof value === 'string' ? value.trim() : '';
  if (!normalized) throw new Error(`${label} is required.`);
  return normalized;
}

function requireHash(value, label) {
  const normalized = requireString(value, label);
  if (!/^[a-f0-9]{64}$/.test(normalized)) {
    throw new Error(`${label} must be a SHA-256 digest.`);
  }
  return normalized;
}

export function buildTrainingPolicyUpdate(input) {
  const parentRolloutHashes = Array.isArray(input?.parentRolloutHashes)
    ? input.parentRolloutHashes.map((hash, index) => requireHash(hash, `parentRolloutHashes[${index}]`))
    : [];
  const artifact = {
    artifactType: 'training_policy_update',
    schemaVersion: 1,
    workloadId: requireString(input?.workloadId, 'workloadId'),
    updateId: requireString(input?.updateId, 'updateId'),
    inputPolicyHash: requireHash(input?.inputPolicyHash, 'inputPolicyHash'),
    outputPolicyHash: requireHash(input?.outputPolicyHash, 'outputPolicyHash'),
    parentRolloutHashes,
    objective: input?.objective,
    metrics: input?.metrics,
    runtime: input?.runtime || null,
    receiptPaths: input?.receiptPaths || null,
    claimBoundary: requireString(input?.claimBoundary, 'claimBoundary'),
  };
  return validateVerifierGuidedArtifact(artifact);
}

export function buildTrainingPolicyCheckpoint(input) {
  const parentArtifactHashes = Array.isArray(input?.parentArtifactHashes)
    ? input.parentArtifactHashes.map((hash, index) => requireHash(hash, `parentArtifactHashes[${index}]`))
    : [];
  const artifact = {
    artifactType: 'training_policy_checkpoint',
    schemaVersion: 1,
    workloadId: requireString(input?.workloadId, 'workloadId'),
    checkpointId: requireString(input?.checkpointId, 'checkpointId'),
    policyHash: requireHash(input?.policyHash, 'policyHash'),
    datasetHash: requireHash(input?.datasetHash, 'datasetHash'),
    parentArtifactHashes,
    adapterPath: requireString(input?.adapterPath, 'adapterPath'),
    checkpointStep: Number(input?.checkpointStep),
    metrics: input?.metrics || {},
    claimBoundary: requireString(input?.claimBoundary, 'claimBoundary'),
  };
  return validateVerifierGuidedArtifact(artifact);
}
