import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';

import {
  assertTrainerArtifactCandidateEntry,
  buildTrainerArtifactImportPlan,
  normalizeGammaTrainerArtifactHandoff,
} from '../../src/experimental/bridge/trainer-artifact-bridge.js';
import { LEGACY_ENUM_VALUE_ALIASES } from '../../src/config/legacy-enum-values.js';
import { sha256Hex } from '../../src/formats/sha256.js';
import { stableSortObject } from '../../src/formats/stable-sort-object.js';

const repoRoot = process.cwd();
const contractPath = path.join(repoRoot, 'tools/policies/nativekd2-bf16-baseline-handoff.json');
const contract = JSON.parse(readFileSync(contractPath, 'utf8'));

const LEGACY_BY_CURRENT = new Map(Object.entries(LEGACY_ENUM_VALUE_ALIASES).map(([legacy, current]) => [current, legacy]));

// Rewrites a plan into the spellings the 0.1.x writer used, to recompute a recorded hash.
function toLegacySpellings(value) {
  if (typeof value === 'string') return LEGACY_BY_CURRENT.get(value) ?? value;
  if (Array.isArray(value)) return value.map(toLegacySpellings);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, toLegacySpellings(entry)]));
  }
  return value;
}

function hashStableJson(value) {
  return sha256Hex(JSON.stringify(stableSortObject(value)));
}

// The handoff normalizes into a valid bridge descriptor.
const descriptor = normalizeGammaTrainerArtifactHandoff(contract);
assert.equal(descriptor.artifact.role, contract.artifactRole);
assert.equal(descriptor.selection.receipt, contract.selection.receipt);

// The import receipt the handoff points at is byte-identical to the recorded identity.
const receiptIdentity = contract.importVerification.receipt;
const receiptPath = path.join(repoRoot, receiptIdentity.path);
const receiptBytes = readFileSync(receiptPath);
assert.equal(statSync(receiptPath).size, receiptIdentity.bytes);
assert.equal(createHash('sha256').update(receiptBytes).digest('hex'), receiptIdentity.sha256);
const receipt = JSON.parse(receiptBytes.toString('utf8'));

// Receipt hashes are self-consistent and chain verification -> plan -> import.
const { receiptHash: verificationHash, ...verificationCore } = receipt.verification;
assert.equal(hashStableJson(verificationCore), verificationHash);
assert.equal(receipt.verification.ok, true);
assert.equal(receipt.verification.bridgeId, descriptor.bridgeId);
assert.equal(receipt.verification.artifactRole, descriptor.artifact.role);

const { receiptHash: importHash, ...importCore } = receipt.importReceipt;
assert.equal(hashStableJson(importCore), importHash);
assert.equal(importHash, contract.importVerification.importReceiptHash);
assert.equal(receipt.importReceipt.identityReceiptHash, verificationHash);

// Admission in the receipts matches what the bridge derives for this descriptor.
// The recorded receipts predate the kebab-case rename and are kept as written: the
// plan reads the Gamma spelling and emits kebab-case, and the recorded plan hash is
// the same plan under the snake_case spellings it was hashed with.
const plan = buildTrainerArtifactImportPlan(descriptor, receipt.verification);
assert.equal(contract.sourceCheckpoint.format, 'huggingface_safetensors');
assert.equal(plan.source.format, 'huggingface-safetensors');
const { planHash: _planHash, ...planCore } = plan;
const recordedPlanCore = toLegacySpellings(planCore);
assert.equal(receipt.importReceipt.importPlanHash, hashStableJson(recordedPlanCore));
for (const admission of [receipt.verification.admission, receipt.importReceipt]) {
  assert.equal(admission.candidateCompetitionAllowed, plan.admission.candidateCompetitionAllowed);
  assert.equal(admission.promotionAllowed, plan.admission.promotionAllowed);
}

// A baseline without a selection receipt cannot enter candidate competition;
// the same handoff with a selection receipt from the named authority can.
if (descriptor.selection.status !== 'selected') {
  assert.throws(() => assertTrainerArtifactCandidateEntry(descriptor), /candidate entry denied/);
}
const selected = normalizeGammaTrainerArtifactHandoff({
  ...contract,
  artifactRole: 'selected_candidate',
  selection: { status: 'gamma_selected', receipt: 'sha256:'.padEnd(71, '0') },
});
assert.doesNotThrow(() => assertTrainerArtifactCandidateEntry(selected));
assert.equal(buildTrainerArtifactImportPlan(selected).admission.candidateCompetitionAllowed, true);
assert.throws(
  () => assertTrainerArtifactCandidateEntry(normalizeGammaTrainerArtifactHandoff({
    ...contract,
    artifactRole: 'selected_candidate',
    authorities: { ...contract.authorities, bf16Selection: 'someone-else' },
    selection: { status: 'gamma_selected', receipt: 'sha256:'.padEnd(71, '0') },
  })),
  /selection authority must be/
);

console.log('nativekd2-trainer-handoff-contract.test: ok');
