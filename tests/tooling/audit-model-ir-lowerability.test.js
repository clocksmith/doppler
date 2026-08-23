import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {
  createLowerabilityAuditReceipt,
  materializeLowerabilityAudit,
} from '../../tools/audit-model-ir-lowerability.js';

const options = {
  modelIRReceiptPath: 'reports/model-ir-v2/glimmer-30b.model-ir-receipt.json',
  vocabularyPath: 'src/config/forge/lowering-vocabularies/heterogeneous-text-v2.json',
  entryPointId: 'text.generate',
  outputPath: 'reports/model-ir-v2/glimmer-30b.lowerability-audit.json',
};
const observed = JSON.parse(await fs.readFile(options.outputPath, 'utf8'));
const recreated = await createLowerabilityAuditReceipt(options);
assert.deepEqual(recreated, observed, 'checked-in lowerability evidence must be deterministic');
const checked = await materializeLowerabilityAudit({ ...options, check: true });
assert.equal(checked.checked, true);
assert.equal(checked.receipt.audit.lowerable, true);
assert.equal(checked.receipt.vocabularyEvidence.scope, 'semantic-contracts-and-manifest-bindings');
assert.equal(checked.receipt.audit.entryPointStatus, 'unlowered');
assert.deepEqual(checked.receipt.audit.requiredPhases, ['prefill', 'decode']);
assert.deepEqual(checked.receipt.audit.unimplementedStateKinds, []);
assert.equal(checked.receipt.audit.component.compatible, true);
assert.equal(checked.receipt.audit.outputHeads[0].compatible, true);
assert.match(JSON.stringify(checked.receipt.audit.blockClasses), /local-attention/);
assert.deepEqual(
  checked.receipt.audit.blockClasses.map((blockClass) => blockClass.compatibleLoweringIds),
  [['portable-local-attention-v2'], ['portable-full-attention-v2']]
);

console.log('✔ audit-model-ir-lowerability.test.js passed');
