import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';

import {
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

const descriptorValidation = validateTrainerArtifactBridgeDescriptor(receipt.descriptor);
assert.equal(descriptorValidation.valid, true, descriptorValidation.errors.join(', '));
assert.equal(receipt.descriptor.artifact.kind, 'peft_adapter');
assert.equal(receipt.descriptor.artifact.role, 'diagnostic_candidate');
assert.equal(receipt.descriptor.selection.authority, 'clocksmith/columbo');
assert.equal(receipt.descriptor.selection.status, 'not_selected');
assert.equal(receipt.descriptor.selection.receipt, null);

const { receiptHash: verificationHash, ...verificationCore } = receipt.verification;
assert.equal(hashStableJson(verificationCore), verificationHash);
assert.equal(receipt.verification.ok, true);
assert.equal(receipt.verification.artifactRole, 'diagnostic_candidate');
assert.equal(receipt.verification.admission.candidateCompetitionAllowed, false);
assert.equal(receipt.verification.admission.promotionAllowed, false);

const rebuiltPlan = buildTrainerArtifactImportPlan(receipt.descriptor, receipt.verification);
assert.equal(rebuiltPlan.planHash, receipt.plan.planHash);
assert.equal(receipt.plan.entrypoint, 'loadLoRAWeights');
assert.equal(receipt.plan.admission.candidateCompetitionAllowed, false);
assert.equal(receipt.plan.admission.promotionAllowed, false);

const { receiptHash: importHash, ...importCore } = receipt.importReceipt;
assert.equal(hashStableJson(importCore), importHash);
assert.equal(receipt.importReceipt.importedIdentity.adapterId,
  'columbo-qwen-peft-r8-attn4-clean943-r04w40-noeval-cpu');
assert.equal(receipt.importReceipt.candidateCompetitionAllowed, false);
assert.equal(receipt.importReceipt.promotionAllowed, false);

console.log('columbo-diagnostic-adapter-handoff-contract.test: ok');
