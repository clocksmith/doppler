import assert from 'node:assert/strict';
import {
  createReleaseToJavaScriptReceipt,
  validateReleaseToJavaScriptReceipt,
} from '../../src/converter/release-to-javascript-receipt.js';

const digest = (character) => `sha256:${character.repeat(64)}`;
const fields = {
  campaignId: 'heterogeneous-model-ir-v2:qwen-test',
  source: {
    checkpointId: 'upstream/test',
    revision: 'abc123',
    publishedAt: '2026-08-20T00:00:00.000Z',
  },
  startedAt: '2026-08-21T00:00:00.000Z',
  completedAt: '2026-08-22T00:00:00.000Z',
  humanInterventions: [
    { id: 'scope', kind: 'campaign-scope', actor: 'reviewer', disposition: 'accepted' },
    { id: 'semantic', kind: 'semantic-decision', actor: 'reviewer', disposition: 'accepted' },
  ],
  unresolvedFacts: [],
  candidates: {
    generated: 3,
    rejected: [{ id: 'unsafe' }, { id: 'unsupported' }],
    accepted: [{ id: 'conservative' }],
  },
  acceptedCode: { revision: 'deadbeef', files: ['src/example.js'], digest: digest('a') },
  qualification: { status: 'passed', packId: 'test.pack', packDigest: digest('b') },
  evidence: [{ kind: 'parity', path: 'reports/parity.json', digest: digest('c') }],
};

const receipt = createReleaseToJavaScriptReceipt(fields);
assert.equal(receipt.elapsed.publicationToSignedPackMs, 172800000);
assert.equal(receipt.elapsed.forgeCampaignMs, 86400000);
assert.equal(receipt.humanAuthoredSemanticDecisions, 1);
assert.equal(validateReleaseToJavaScriptReceipt(receipt).ok, true);

const drifted = structuredClone(receipt);
drifted.candidates.generated = 4;
const validation = validateReleaseToJavaScriptReceipt(drifted);
assert.equal(validation.ok, false);
assert.match(validation.errors.join('; '), /candidates\.generated|receiptDigest mismatch/);

console.log('✔ release-to-javascript-receipt.test.js passed');
