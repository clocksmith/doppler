import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import { evaluateTranslationArtifactCompetition } from '../../src/tooling/translation-artifact-competition.js';

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

const policy = readJson('tools/policies/translation-artifact-competition.json');
const handoffPath = 'tools/policies/nativekd2-bf16-baseline-handoff.json';
const handoffBytes = readFileSync(handoffPath);
const handoff = JSON.parse(handoffBytes.toString('utf8'));
const handoffSha256 = createHash('sha256').update(handoffBytes).digest('hex');
const verification = readJson('docs/status/nativekd2-bf16-baseline-import-receipt-2026-07-13.json');
const committed = readJson('docs/status/translation-artifact-competition-readiness-2026-07-13.json');

// The committed readiness receipt is the evaluator's output over the committed inputs.
const replayed = evaluateTranslationArtifactCompetition({
  policy,
  handoff,
  handoffSha256,
  verificationReceipt: verification,
});
assert.deepEqual(replayed, committed);

// The observed source is the handoff the evaluator was given.
assert.equal(committed.observedSource.artifactRole, handoff.artifactRole);
assert.equal(committed.observedSource.selectionReceipt, handoff.selection.receipt);
assert.equal(committed.observedSource.handoffSha256, handoffSha256);

// Admission is monotonic and the decision names the strongest admitted stage.
function assertAdmissionCoherent(receipt) {
  const { artifactGenerationAllowed, artifactComparisonAllowed, promotionSubmissionAllowed } = receipt.admission;
  if (promotionSubmissionAllowed) assert.equal(artifactComparisonAllowed, true);
  if (artifactComparisonAllowed) assert.equal(artifactGenerationAllowed, true);
  const expectedDecision = promotionSubmissionAllowed
    ? 'promotion_submission_allowed'
    : artifactComparisonAllowed
      ? 'artifact_comparison_allowed'
      : artifactGenerationAllowed ? 'artifact_generation_allowed' : 'blocked';
  assert.equal(receipt.decision, expectedDecision);
  assert.equal(artifactGenerationAllowed, receipt.blockers.length === 0,
    'artifact generation is allowed exactly when no blocker remains');
}
assertAdmissionCoherent(committed);

// A source without a selection receipt cannot enter the competition.
if (handoff.selection.status !== 'selected') {
  assert.equal(committed.admission.artifactGenerationAllowed, false);
  assert.ok(committed.blockers.includes('gamma_source_not_selected_candidate'));
}

// Failure paths: missing handoff blocks, malformed policy fails fast.
{
  const absent = evaluateTranslationArtifactCompetition({ policy });
  assert.equal(absent.observedSource, null);
  assert.ok(absent.blockers.includes('gamma_selected_handoff_absent'));
  assertAdmissionCoherent(absent);

  assert.throws(
    () => evaluateTranslationArtifactCompetition({ policy: { ...policy, schemaVersion: 2 }, handoff }),
    /schemaVersion must be 1/
  );
  assert.throws(
    () => evaluateTranslationArtifactCompetition({
      policy: { ...policy, artifactLanes: [...policy.artifactLanes].reverse() },
      handoff,
    }),
    /lanes are required in order/
  );
}

console.log('translation-artifact-competition-contract.test: ok');
