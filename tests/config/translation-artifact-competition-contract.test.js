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
const verification = readJson('docs/status/nativekd2-bf16-baseline-import-receipt-2026-07-13.json');
const committed = readJson('docs/status/translation-artifact-competition-readiness-2026-07-13.json');
const replayed = evaluateTranslationArtifactCompetition({
  policy,
  handoff,
  handoffSha256: createHash('sha256').update(handoffBytes).digest('hex'),
  verificationReceipt: verification,
});

assert.deepEqual(replayed, committed);
assert.equal(committed.decision, 'blocked');
assert.equal(committed.observedSource.artifactRole, 'diagnostic_baseline');
assert.equal(committed.observedSource.selectionReceipt, null);
assert.equal(committed.admission.artifactGenerationAllowed, false);
assert.equal(committed.admission.artifactComparisonAllowed, false);
assert.equal(committed.admission.promotionSubmissionAllowed, false);
assert.ok(committed.blockers.includes('gamma_source_not_selected_candidate'));
assert.ok(committed.blockers.includes('gamma_bf16_selection_receipt_absent'));

console.log('translation-artifact-competition-contract.test: ok');
