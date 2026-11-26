import {
  assertTrainerArtifactCandidateEntry,
  normalizeGammaTrainerArtifactHandoff,
} from '../experimental/bridge/trainer-artifact-bridge.js';
import { canonicalizeLegacyEnumValue, canonicalizeLegacyEnumValues } from '../config/legacy-enum-values.js';
import { isPlainObject } from '../formats/plain-object.js';
import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';

export const TRANSLATION_ARTIFACT_COMPETITION_SCHEMA_ID =
  'doppler.translation-artifact-competition-readiness/v1';

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const REQUIRED_LANES = Object.freeze(['q4k', 'selective-f16', 'qat']);

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function hashStableJson(value) {
  return sha256Hex(stableJson(value));
}

// Lane ids written before the kebab-case rename (selective_f16) are read as
// their kebab-case form; receipts emit only the kebab-case form.
function canonicalizeLanes(lanes) {
  return lanes.map((lane) => (
    isPlainObject(lane) ? { ...lane, laneId: canonicalizeLegacyEnumValue(lane.laneId) } : lane
  ));
}

function requirePolicy(rawPolicy) {
  if (!isPlainObject(rawPolicy)) {
    throw new Error('translation artifact competition: policy must be an object.');
  }
  const policy = Array.isArray(rawPolicy.artifactLanes)
    ? { ...rawPolicy, artifactLanes: canonicalizeLanes(rawPolicy.artifactLanes) }
    : rawPolicy;
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
  if (canonicalizeLegacyEnumValue(policy.state) === 'blocked-awaiting-gamma-selection') {
    blockers.push('competition-policy-not-bound-to-gamma-selected-source');
  }
  if (!SHA256_PATTERN.test(String(handoffSha256 || ''))) {
    blockers.push('gamma-selected-handoff-sha256-absent');
  } else if (source.handoffSha256 !== handoffSha256) {
    blockers.push('gamma-selected-handoff-sha256-mismatch');
  }
  if (source.selectionReceipt !== descriptor.selection.receipt) {
    blockers.push('gamma-selection-receipt-binding-mismatch');
  }
  if (source.selectedCheckpointSha256 !== descriptor.baseModel.checkpointSha256) {
    blockers.push('gamma-selected-checkpoint-binding-mismatch');
  }
  if (!verificationReceipt) {
    blockers.push('selected-source-identity-verification-absent-or-invalid');
  } else {
    if (verificationReceipt.ok !== true) {
      blockers.push('selected-source-identity-verification-failed');
    }
    if (verificationReceipt.bridgeId !== descriptor.bridgeId) {
      blockers.push('selected-source-identity-bridge-mismatch');
    }
    if (verificationReceipt.artifactRole !== 'selected_candidate') {
      blockers.push('selected-source-identity-role-mismatch');
    }
    if (source.identityVerificationReceiptHash !== verificationReceipt.receiptHash) {
      blockers.push('selected-source-identity-receipt-binding-mismatch');
    }
  }
  return blockers;
}

export function evaluateTranslationArtifactCompetition(options = {}) {
  const policy = requirePolicy(options.policy);
  // The policy file is digest-bound and keeps its snake_case spellings; they are read
  // throughout 0.x and emitted in kebab-case.
  const blockers = new Set(Array.isArray(policy.blockers) ? policy.blockers.map(canonicalizeLegacyEnumValue) : []);
  let descriptor = null;
  let verificationReceipt = null;

  if (!isPlainObject(options.handoff)) {
    blockers.add('gamma-selected-handoff-absent');
  } else {
    try {
      descriptor = normalizeGammaTrainerArtifactHandoff(options.handoff);
      assertTrainerArtifactCandidateEntry(descriptor);
    } catch {
      blockers.add('gamma-source-not-selected-candidate');
      blockers.add('gamma-source-selection-contract-invalid');
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
  if (sourceReady && !lanesEvaluated) blockers.add('artifact-lane-evidence-incomplete');
  if (lanesEvaluated && selectedLanes.length !== 1) blockers.add('artifact-lane-selection-not-unique');

  const artifactGenerationAllowed = sourceReady;
  const artifactComparisonAllowed = sourceReady && lanesEvaluated;
  const promotionSubmissionAllowed = artifactComparisonAllowed
    && selectedLanes.length === 1
    && policy.state === 'terminal';
  const receiptCore = {
    schema: TRANSLATION_ARTIFACT_COMPETITION_SCHEMA_ID,
    contractId: policy.contractId,
    policyState: canonicalizeLegacyEnumValue(policy.state),
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
      ? 'promotion-submission-allowed'
      : artifactComparisonAllowed
        ? 'artifact-comparison-allowed'
        : artifactGenerationAllowed
          ? 'artifact-generation-allowed'
          : 'blocked',
    blockers: [...blockers].sort(),
  };
  return { ...receiptCore, receiptHash: hashStableJson(receiptCore) };
}

// Reads a recorded readiness receipt. The recorded receiptHash covers the
// receipt exactly as written, so it is verified before lane ids are
// canonicalized; the returned receipt keeps the recorded hash as its
// attestation of the original bytes.
export function readTranslationArtifactCompetitionReceipt(recorded) {
  if (!isPlainObject(recorded)) {
    throw new Error('translation artifact competition: receipt must be an object.');
  }
  const { receiptHash, ...core } = recorded;
  if (!SHA256_PATTERN.test(String(receiptHash || '')) || hashStableJson(core) !== receiptHash) {
    throw new Error('translation artifact competition: receipt hash does not match the recorded receipt.');
  }
  // The recorded receipt keeps its original spelling and hash; the returned copy is kebab-case.
  return canonicalizeLegacyEnumValues(Array.isArray(recorded.lanes)
    ? { ...recorded, lanes: canonicalizeLanes(recorded.lanes) }
    : recorded);
}
