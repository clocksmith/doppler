import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';

import {
  assertTrainerArtifactCandidateEntry,
  buildTrainerArtifactImportPlan,
  validateTrainerArtifactBridgeDescriptor,
} from '../../src/experimental/bridge/trainer-artifact-bridge.js';
import { sha256Hex } from '../../src/utils/sha256.js';
import { stableSortObject } from '../../src/utils/stable-sort-object.js';

const repoRoot = process.cwd();
const receiptPath = path.join(
  repoRoot,
  'docs/status/columbo-qwen-cpu-diagnostic-adapter-import-receipt-2026-07-13.json'
);
const receipt = JSON.parse(readFileSync(receiptPath, 'utf8'));

function hashStableJson(value) {
  return sha256Hex(JSON.stringify(stableSortObject(value)));
}

// The recorded descriptor is valid under the bridge's own validator.
const descriptorValidation = validateTrainerArtifactBridgeDescriptor(receipt.descriptor);
assert.equal(descriptorValidation.valid, true, descriptorValidation.errors.join(', '));
const descriptor = descriptorValidation.descriptor;

// Verification receipt hash is self-consistent and binds to this descriptor.
const { receiptHash: verificationHash, ...verificationCore } = receipt.verification;
assert.equal(hashStableJson(verificationCore), verificationHash);
assert.equal(receipt.verification.ok, true);
assert.equal(receipt.verification.bridgeId, descriptor.bridgeId);
assert.equal(receipt.verification.artifactRole, descriptor.artifact.role);

// The recorded plan is exactly what the bridge rebuilds from descriptor + verification.
const rebuiltPlan = buildTrainerArtifactImportPlan(receipt.descriptor, receipt.verification);
assert.deepEqual(receipt.plan, rebuiltPlan);

// The import receipt chains to the plan and verification, and repeats the derived admission.
const { receiptHash: importHash, ...importCore } = receipt.importReceipt;
assert.equal(hashStableJson(importCore), importHash);
assert.equal(receipt.importReceipt.importPlanHash, rebuiltPlan.planHash);
assert.equal(receipt.importReceipt.identityReceiptHash, verificationHash);
assert.equal(receipt.importReceipt.importedIdentity.adapterId, path.posix.basename(descriptor.artifact.rootPath));
assert.equal(receipt.importReceipt.importedIdentity.baseModel, descriptor.baseModel.modelId);
for (const admission of [receipt.verification.admission, receipt.importReceipt]) {
  assert.equal(admission.candidateCompetitionAllowed, rebuiltPlan.admission.candidateCompetitionAllowed);
  assert.equal(admission.promotionAllowed, rebuiltPlan.admission.promotionAllowed);
}

// An unselected diagnostic adapter cannot enter candidate competition.
if (descriptor.selection.status !== 'selected') {
  assert.equal(rebuiltPlan.admission.candidateCompetitionAllowed, false);
  assert.throws(() => assertTrainerArtifactCandidateEntry(receipt.descriptor), /candidate entry denied/);
}

// Failure paths: a failing verification cannot produce a plan; a broken descriptor is rejected.
assert.throws(
  () => buildTrainerArtifactImportPlan(receipt.descriptor, { ...receipt.verification, ok: false }),
  /requires a passing verification receipt/
);
const broken = structuredClone(receipt.descriptor);
delete broken.artifact.kind;
assert.equal(validateTrainerArtifactBridgeDescriptor(broken).valid, false);

console.log('columbo-diagnostic-adapter-handoff-contract.test: ok');
