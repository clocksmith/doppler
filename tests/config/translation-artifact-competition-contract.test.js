import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import {
  evaluateTranslationArtifactCompetition,
  readTranslationArtifactCompetitionReceipt,
} from '../../src/tooling/translation-artifact-competition.js';

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

const policy = readJson('tools/policies/translation-artifact-competition.json');
const handoffPath = 'tools/policies/nativekd2-bf16-baseline-handoff.json';
const handoffBytes = readFileSync(handoffPath);
const handoff = JSON.parse(handoffBytes.toString('utf8'));
const handoffSha256 = createHash('sha256').update(handoffBytes).digest('hex');
const verification = readJson('docs/status/nativekd2-bf16-baseline-import-receipt-2026-07-13.json');
// The recorded receipt predates the kebab-case lane rename (selective_f16) and is
// kept byte-for-byte; the reader verifies its hash and canonicalizes lane ids.
const recorded = readJson('docs/status/translation-artifact-competition-readiness-2026-07-13.json');
const committed = readTranslationArtifactCompetitionReceipt(recorded);
assert.equal(recorded.lanes[1].laneId, 'selective_f16');
assert.equal(committed.lanes[1].laneId, 'selective-f16');
assert.throws(
  () => readTranslationArtifactCompetitionReceipt({ ...recorded, decision: 'artifact-generation-allowed' }),
  /receipt hash does not match/
);

// The committed readiness receipt is the evaluator's output over the committed inputs.
const replayed = evaluateTranslationArtifactCompetition({
  policy,
  handoff,
  handoffSha256,
  verificationReceipt: verification,
});
const { receiptHash: _replayedHash, ...replayedCore } = replayed;
const { receiptHash: _recordedHash, ...committedCore } = committed;
assert.deepEqual(replayedCore, committedCore);

// A policy still carrying the legacy lane spelling is read in kebab-case.
const legacyPolicy = structuredClone(policy);
legacyPolicy.artifactLanes[1].laneId = 'selective_f16';
assert.deepEqual(
  evaluateTranslationArtifactCompetition({ policy: legacyPolicy, handoff, handoffSha256, verificationReceipt: verification }),
  replayed
);

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
    ? 'promotion-submission-allowed'
    : artifactComparisonAllowed
      ? 'artifact-comparison-allowed'
      : artifactGenerationAllowed ? 'artifact-generation-allowed' : 'blocked';
  assert.equal(receipt.decision, expectedDecision);
  assert.equal(artifactGenerationAllowed, receipt.blockers.length === 0,
    'artifact generation is allowed exactly when no blocker remains');
}
assertAdmissionCoherent(committed);

// A source without a selection receipt cannot enter the competition.
if (handoff.selection.status !== 'selected') {
  assert.equal(committed.admission.artifactGenerationAllowed, false);
  assert.ok(committed.blockers.includes('gamma-source-not-selected-candidate'));
}

// Failure paths: missing handoff blocks, malformed policy fails fast.
{
  const absent = evaluateTranslationArtifactCompetition({ policy });
  assert.equal(absent.observedSource, null);
  assert.ok(absent.blockers.includes('gamma-selected-handoff-absent'));
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
