import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { hashEvidenceValue } from '../../src/client/model-host/model-evidence.js';
const evidencePath = 'artifacts/create-generation-qualification/browser-constraint-recovery.json';
const bytes = fs.readFileSync(evidencePath);
const record = JSON.parse(bytes);
assert.equal(record.result.status, 'finished');
assert.match(record.device.userAgent, /Chrome/);
assert.equal(record.device.info.vendor, 'intel');
const checks = [];
for (const [field, count] of [['evidence',3],['repeat',5],['recovered',2]]) {
  const result = record.result[field];
  assert.deepEqual(JSON.parse(result.outputText), {count});
  assert.equal(result.stats.stopReason, 'stop-token');
  assert.equal(result.stats.decodeMode, 'single_token');
  assert.equal(result.generationConfig.logitMaskIdentity.id, 'json-object-syntax-v1');
  assert.equal(result.generationConfigHash, await hashEvidenceValue(result.generationConfig));
  assert.equal(result.transcriptHash, await hashEvidenceValue(result.transcript));
  assert.equal(result.runtimeProfileHash, await hashEvidenceValue(result.runtimeProfile));
  assert.equal(result.backendIdentityHash, await hashEvidenceValue(result.backendIdentity));
  assert.equal(result.resolution.resolvedExecutionId, await hashEvidenceValue(result.executionIdentity));
  checks.push({field,expected:{count},actual:JSON.parse(result.outputText),transcriptHash:result.transcriptHash});
}
assert.equal(record.result.cancelled.stats.stopReason, 'aborted');
assert.equal(record.result.maskFailure.message, 'qualification malformed constraint');
const report = {schema:'simulatte.generationBrowserDiagnostic.v1',capturedAt:new Date().toISOString(),
  evidencePath,evidenceSha256:createHash('sha256').update(bytes).digest('hex'),
  evidenceLayer:'actual-browser-component',checks,cancellationObserved:true,constraintFailureRecoveryObserved:true,
  qualifiedForCreate:false,remaining:['full linguistic and semantic schema fidelity','model and graphics budgets in Create','device loss and recovery','public dependency pin migration','application and held-out composition validation']};
fs.writeFileSync('artifacts/create-generation-qualification/browser-verification.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
