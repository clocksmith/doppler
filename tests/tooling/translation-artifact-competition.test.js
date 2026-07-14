import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import { normalizeGammaTrainerArtifactHandoff } from '../../src/experimental/bridge/trainer-artifact-bridge.js';
import { evaluateTranslationArtifactCompetition } from '../../src/tooling/translation-artifact-competition.js';
import { sha256Hex } from '../../src/utils/sha256.js';
import { stableSortObject } from '../../src/utils/stable-sort-object.js';

const policy = JSON.parse(readFileSync('tools/policies/translation-artifact-competition.json', 'utf8'));
const baseline = JSON.parse(readFileSync('tools/policies/nativekd2-bf16-baseline-handoff.json', 'utf8'));

function hashStableJson(value) {
  return sha256Hex(JSON.stringify(stableSortObject(value)));
}

const absent = evaluateTranslationArtifactCompetition({ policy });
assert.equal(absent.decision, 'blocked');
assert.equal(absent.admission.artifactGenerationAllowed, false);
assert.ok(absent.blockers.includes('gamma_selected_handoff_absent'));

const baselineResult = evaluateTranslationArtifactCompetition({
  policy,
  handoff: baseline,
  handoffSha256: createHash('sha256')
    .update(readFileSync('tools/policies/nativekd2-bf16-baseline-handoff.json'))
    .digest('hex'),
});
assert.equal(baselineResult.decision, 'blocked');
assert.equal(baselineResult.admission.artifactGenerationAllowed, false);
assert.ok(baselineResult.blockers.includes('gamma_source_not_selected_candidate'));

const selectedHandoff = structuredClone(baseline);
selectedHandoff.contractId = 'gamma.translation.selected-unit.v1';
selectedHandoff.state = 'gamma_selected_bf16';
selectedHandoff.artifactRole = 'selected_candidate';
selectedHandoff.conversionLineage.status = 'selected_checkpoint_awaiting_conversion';
selectedHandoff.conversionLineage.futureUse = 'gamma_selected_checkpoint_artifact_competition';
selectedHandoff.evaluationInputs.populationRole = 'gamma_selection_evidence';
selectedHandoff.selection.status = 'gamma_selected';
selectedHandoff.selection.receipt = 'gamma.selection-receipt/v1:unit';
const descriptor = normalizeGammaTrainerArtifactHandoff(selectedHandoff);
const selectedHandoffSha256 = hashStableJson(selectedHandoff);
const verificationCore = {
  schema: 'doppler.trainer-artifact-handoff-verification/v1',
  bridgeId: descriptor.bridgeId,
  sourceContractId: descriptor.sourceContractId,
  artifactKind: descriptor.artifact.kind,
  artifactRole: descriptor.artifact.role,
  ok: true,
  artifactIdentitySha256: '1'.repeat(64),
  checks: [],
  files: [],
  architecture: { ok: true, fields: [], errors: [] },
};
const verification = {
  ...verificationCore,
  receiptHash: hashStableJson(verificationCore),
};
const selectedPolicy = structuredClone(policy);
selectedPolicy.state = 'source_selected_artifact_generation';
selectedPolicy.gammaSource = {
  handoffPath: 'gamma-selected-handoff.json',
  handoffSha256: selectedHandoffSha256,
  selectionReceipt: descriptor.selection.receipt,
  selectedCheckpointSha256: descriptor.baseModel.checkpointSha256,
  identityVerificationReceiptHash: verification.receiptHash,
};
selectedPolicy.blockers = [];
selectedPolicy.admission.artifactGenerationAllowed = true;

const selectedResult = evaluateTranslationArtifactCompetition({
  policy: selectedPolicy,
  handoff: selectedHandoff,
  handoffSha256: selectedHandoffSha256,
  verificationReceipt: verification,
});
assert.equal(selectedResult.decision, 'artifact_generation_allowed');
assert.equal(selectedResult.admission.artifactGenerationAllowed, true);
assert.equal(selectedResult.admission.artifactComparisonAllowed, false);
assert.equal(selectedResult.admission.promotionSubmissionAllowed, false);
assert.deepEqual(selectedResult.blockers, ['artifact_lane_evidence_incomplete']);

console.log('translation-artifact-competition.test: ok');
