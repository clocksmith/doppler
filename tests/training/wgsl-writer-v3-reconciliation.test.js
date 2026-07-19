import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { hashWgslSemanticEvidenceValue } from '../../src/tooling/wgsl-repair-semantic-gate.js';
import { reconcileWgslWriterV3Campaign } from '../../tools/reconcile-wgsl-writer-v3-campaign.js';

const EXPECTED_SCHEMA_SHA256 = '5262a2ed29dd97d163c49f21ab69b54103dc524c68959dc0561defb128fdc038';

const receiptPath = 'docs/status/wgsl-writer-v3-campaign-reconciliation-2026-07-19.json';
const receipt = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
const { receiptHash, ...core } = receipt;

assert.equal(receipt.schema, 'doppler.wgsl-writer-v3-campaign-reconciliation/v1');
assert.equal(receipt.exposureLedgerContract.schemaSha256, EXPECTED_SCHEMA_SHA256);
assert.equal(receipt.originalGate.status, 'reference_qualified_corpus_materialization_blocked');
assert.equal(receipt.originalGate.trainingAllowed, false);
assert.equal(receipt.laterPolicies.length, 4);
assert.equal(receipt.laterPolicies.every((policy) => policy.role === 'development'), true);
assert.equal(receipt.resolution.experimentState, 'frozen');
assert.equal(receipt.resolution.promotionAuthority, false);
assert.equal(receipt.nextRequiredTransition.to, 'materialized');
assert.equal(receiptHash, hashWgslSemanticEvidenceValue(core));

assert.equal(receipt.originalGate.sha256, '4af9d8cf8791b6c70250cee6ddf8000925f425361c636e36a80e8143305b43bc');
assert.equal(receipt.originalGate.repositoryRevision, '9f0ef19dbe5fd9b1c69e1cd7002005c7d3ea8d6d');
for (const binding of receipt.laterPolicies) {
  const observed = createHash('sha256').update(await fs.readFile(binding.path)).digest('hex');
  assert.equal(observed, binding.sha256, binding.path);
}

const temporaryRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-v3-reconcile-'));
const regeneratedPath = path.join(temporaryRoot, 'receipt.json');
const regenerated = await reconcileWgslWriterV3Campaign({ outputPath: regeneratedPath });
assert.equal(regenerated.receipt.receiptHash, receipt.receiptHash);
