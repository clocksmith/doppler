import assert from 'node:assert/strict';
import { sha256Hex } from '../../src/utils/sha256.js';
import { stableSortObject } from '../../src/utils/stable-sort-object.js';
import {
  createReleaseToJavaScriptReceipt,
  validateReleaseToJavaScriptReceipt,
} from '../../src/converter/release-to-javascript-receipt.js';

const digest = (character) => `sha256:${character.repeat(64)}`;
const hashValue = (value) => `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
const acceptedFiles = [{ path: 'src/example.js', digest: digest('d') }];
const fields = {
  campaignId: 'heterogeneous-model-ir-v2:qwen-test',
  source: {
    checkpointId: 'upstream/test',
    revision: 'abc123',
    publicationTimestampDisposition: 'observed',
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
  acceptedCode: {
    revision: 'deadbeef',
    files: acceptedFiles,
    digest: hashValue({ revision: 'deadbeef', files: acceptedFiles }),
  },
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

const driftedCode = structuredClone(receipt);
driftedCode.acceptedCode.files[0].digest = digest('e');
const codeValidation = validateReleaseToJavaScriptReceipt(driftedCode);
assert.equal(codeValidation.ok, false);
assert.match(codeValidation.errors.join('; '), /acceptedCode\.digest mismatch|receiptDigest mismatch/);

const invalidCodeFile = structuredClone(receipt);
invalidCodeFile.acceptedCode.files[0] = { digest: digest('f') };
const invalidCodeValidation = validateReleaseToJavaScriptReceipt(invalidCodeFile);
assert.equal(invalidCodeValidation.ok, false);
assert.match(invalidCodeValidation.errors.join('; '), /acceptedCode\.files\[0\]\.path|receiptDigest mismatch/);

const unresolvedPublication = createReleaseToJavaScriptReceipt({
  ...structuredClone(fields),
  campaignId: 'heterogeneous-model-ir-v2:unresolved-publication',
  source: {
    checkpointId: 'upstream/unpublished-date',
    revision: 'def456',
    publicationTimestampDisposition: 'unresolved',
    publishedAt: null,
  },
  unresolvedFacts: [{
    id: 'source-publication-timestamp',
    disposition: 'unresolved',
    evidence: 'Pinned source snapshot contains no authoritative publication timestamp.',
  }],
});
assert.equal(unresolvedPublication.elapsed.publicationToSignedPackMs, null);
assert.equal(unresolvedPublication.elapsed.forgeCampaignMs, 86400000);
assert.equal(validateReleaseToJavaScriptReceipt(unresolvedPublication).ok, true);

const inventedPublication = structuredClone(unresolvedPublication);
inventedPublication.source.publishedAt = '2026-08-20T00:00:00.000Z';
const inventedValidation = validateReleaseToJavaScriptReceipt(inventedPublication);
assert.equal(inventedValidation.ok, false);
assert.match(inventedValidation.errors.join('; '), /publishedAt must be null|receiptDigest mismatch/);

const missingPublicationFact = structuredClone(unresolvedPublication);
missingPublicationFact.unresolvedFacts = [];
const unresolvedValidation = validateReleaseToJavaScriptReceipt(missingPublicationFact);
assert.equal(unresolvedValidation.ok, false);
assert.match(unresolvedValidation.errors.join('; '), /source-publication-timestamp|receiptDigest mismatch/);

console.log('✔ release-to-javascript-receipt.test.js passed');
