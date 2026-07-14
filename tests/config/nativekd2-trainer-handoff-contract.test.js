import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const repoRoot = process.cwd();
const contractPath = path.join(
  repoRoot,
  'tools/policies/nativekd2-bf16-baseline-handoff.json'
);
const contract = JSON.parse(readFileSync(contractPath, 'utf8'));

assert.equal(contract.state, 'baseline_frozen_not_selected');
assert.equal(contract.artifactRole, 'diagnostic_baseline');
assert.equal(contract.authorities.bf16Selection, 'clocksmith/gamma');
assert.equal(contract.selection.status, 'not_selected');
assert.equal(contract.selection.receipt, null);
assert.equal(contract.parity.machineryStatus, 'import_verified');
assert.equal(contract.importVerification.status, 'verified');

const receiptIdentity = contract.importVerification.receipt;
const receiptPath = path.join(repoRoot, receiptIdentity.path);
const receiptBytes = readFileSync(receiptPath);
assert.equal(statSync(receiptPath).size, receiptIdentity.bytes);
assert.equal(createHash('sha256').update(receiptBytes).digest('hex'), receiptIdentity.sha256);

const receipt = JSON.parse(receiptBytes.toString('utf8'));
assert.equal(receipt.verification.ok, true);
assert.equal(receipt.verification.artifactRole, 'diagnostic_baseline');
assert.equal(receipt.verification.admission.candidateCompetitionAllowed, false);
assert.equal(receipt.verification.admission.promotionAllowed, false);
assert.equal(receipt.importReceipt.receiptHash, contract.importVerification.importReceiptHash);
assert.equal(receipt.importReceipt.importedIdentity.sourceKind, 'safetensors');
assert.equal(receipt.importReceipt.candidateCompetitionAllowed, false);
assert.equal(receipt.importReceipt.promotionAllowed, false);

console.log('nativekd2-trainer-handoff-contract.test: ok');
