import {
  assertTrainerArtifactCandidateEntry,
  normalizeGammaTrainerArtifactHandoff,
} from '../experimental/bridge/trainer-artifact-bridge.js';
import { isPlainObject } from '../utils/plain-object.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const TRANSLATION_ARTIFACT_COMPETITION_SCHEMA_ID =
  'doppler.translation-artifact-competition-readiness/v1';

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const REQUIRED_LANES = Object.freeze(['q4k', 'selective_f16', 'qat']);

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function hashStableJson(value) {
  return sha256Hex(stableJson(value));
}

function requirePolicy(policy) {
  if (!isPlainObject(policy)) {
    throw new Error('translation artifact competition: policy must be an object.');
  }
  if (policy.schemaVersion !== 1) {
    throw new Error('translation artifact competition: policy schemaVersion must be 1.');
  }
  if (policy.contractId !== 'doppler.translation.selected-bf16-artifact-competition.v1') {
    throw new Error('translation artifact competition: policy contractId is unsupported.');
  }
  if (policy.owner !== 'clocksmith/doppler') {
    throw new Error('translation artifact competition: policy owner must be clocksmith/doppler.');
  }
  const lanes = Array.isArray(policy.artifactLanes) ? policy.artifactLanes : [];
  if (stableJson(lanes.map((lane) => lane?.laneId)) !== stableJson(REQUIRED_LANES)) {
    throw new Error('translation artifact competition: Q4K, selective-F16, and QAT lanes are required in order.');
  }
  return policy;
}

function normalizeVerificationReceipt(value) {
  if (!isPlainObject(value)) return null;
  const receipt = isPlainObject(value.verification) ? value.verification : value;
  if (!isPlainObject(receipt)) return null;
  const { receiptHash, ...core } = receipt;
  if (!SHA256_PATTERN.test(String(receiptHash || ''))) return null;
  if (hashStableJson(core) !== receiptHash) return null;
  return receipt;
}

function sourceBindingBlockers({ policy, descriptor, handoffSha256, verificationReceipt }) {
  const blockers = [];
  const source = policy.gammaSource || {};
  if (policy.state === 'blocked_awaiting_gamma_selection') {
    blockers.push('competition_policy_not_bound_to_gamma_selected_source');
  }
  if (!SHA256_PATTERN.test(String(handoffSha256 || ''))) {
    blockers.push('gamma_selected_handoff_sha256_absent');
  } else if (source.handoffSha256 !== handoffSha256) {
    blockers.push('gamma_selected_handoff_sha256_mismatch');
  }
  if (source.selectionReceipt !== descriptor.selection.receipt) {
    blockers.push('gamma_selection_receipt_binding_mismatch');
  }
  if (source.selectedCheckpointSha256 !== descriptor.baseModel.checkpointSha256) {
    blockers.push('gamma_selected_checkpoint_binding_mismatch');
  }
  if (!verificationReceipt) {
    blockers.push('selected_source_identity_verification_absent_or_invalid');
  } else {
    if (verificationReceipt.ok !== true) {
      blockers.push('selected_source_identity_verification_failed');
    }
    if (verificationReceipt.bridgeId !== descriptor.bridgeId) {
      blockers.push('selected_source_identity_bridge_mismatch');
    }
    if (verificationReceipt.artifactRole !== 'selected_candidate') {
      blockers.push('selected_source_identity_role_mismatch');
    }
    if (source.identityVerificationReceiptHash !== verificationReceipt.receiptHash) {
      blockers.push('selected_source_identity_receipt_binding_mismatch');
    }
  }
  return blockers;
}

export function evaluateTranslationArtifactCompetition(options = {}) {
  const policy = requirePolicy(options.policy);
  const blockers = new Set(Array.isArray(policy.blockers) ? policy.blockers : []);
  let descriptor = null;
  let verificationReceipt = null;

  if (!isPlainObject(options.handoff)) {
    blockers.add('gamma_selected_handoff_absent');
  } else {
    try {
      descriptor = normalizeGammaTrainerArtifactHandoff(options.handoff);
      assertTrainerArtifactCandidateEntry(descriptor);
    } catch {
      blockers.add('gamma_source_not_selected_candidate');
      blockers.add('gamma_source_selection_contract_invalid');
    }
  }

  if (descriptor) {
    verificationReceipt = normalizeVerificationReceipt(options.verificationReceipt);
    for (const blocker of sourceBindingBlockers({
      policy,
      descriptor,
      handoffSha256: options.handoffSha256,
      verificationReceipt,
    })) blockers.add(blocker);
  }

  const sourceReady = blockers.size === 0;
  const lanes = policy.artifactLanes.map((lane) => ({
    laneId: lane.laneId,
    status: lane.status,
    artifactReceipt: lane.artifactReceipt,
  }));
  const lanesEvaluated = lanes.every(
    (lane) => ['evaluated', 'rejected', 'selected'].includes(lane.status) && lane.artifactReceipt
  );
  const selectedLanes = lanes.filter((lane) => lane.status === 'selected');
  if (sourceReady && !lanesEvaluated) blockers.add('artifact_lane_evidence_incomplete');
  if (lanesEvaluated && selectedLanes.length !== 1) blockers.add('artifact_lane_selection_not_unique');

  const artifactGenerationAllowed = sourceReady;
  const artifactComparisonAllowed = sourceReady && lanesEvaluated;
  const promotionSubmissionAllowed = artifactComparisonAllowed
    && selectedLanes.length === 1
    && policy.state === 'terminal';
  const receiptCore = {
    schema: TRANSLATION_ARTIFACT_COMPETITION_SCHEMA_ID,
    contractId: policy.contractId,
    policyState: policy.state,
    gammaSelectionAuthority: policy.gammaContract?.selectionAuthority ?? null,
    observedSource: descriptor ? {
      bridgeId: descriptor.bridgeId,
      artifactRole: descriptor.artifact.role,
      checkpointSha256: descriptor.baseModel.checkpointSha256,
      selectionReceipt: descriptor.selection.receipt,
      handoffSha256: options.handoffSha256 ?? null,
      identityVerificationReceiptHash: verificationReceipt?.receiptHash ?? null,
    } : null,
    lanes,
    admission: {
      artifactGenerationAllowed,
      artifactComparisonAllowed,
      promotionSubmissionAllowed,
    },
    decision: promotionSubmissionAllowed
      ? 'promotion_submission_allowed'
      : artifactComparisonAllowed
        ? 'artifact_comparison_allowed'
        : artifactGenerationAllowed
          ? 'artifact_generation_allowed'
          : 'blocked',
    blockers: [...blockers].sort(),
  };
  return { ...receiptCore, receiptHash: hashStableJson(receiptCore) };
}
