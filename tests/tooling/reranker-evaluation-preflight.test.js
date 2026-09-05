import assert from 'node:assert/strict';
import { qualifyRerankerElectron } from '../../tools/qualify-reranker-electron.js';
import { buildRerankerEvaluationPack } from '../../tools/build-reranker-evaluation-pack.js';

await assert.rejects(qualifyRerankerElectron({ mode: 'pack', fault: { kind: 'unknown' } }), /Unsupported qualification fault/);
await assert.rejects(qualifyRerankerElectron({ mode: 'model', fault: { kind: 'device-loss' } }), /Unsupported qualification fault/);
await assert.rejects(qualifyRerankerElectron({ mode: 'model' }), /requires policyPath/);
const config = { mode: 'pack', policyPath: 'missing-policy', referencePath: 'missing-reference',
  modelDir: 'missing-model', packageRoot: 'missing-package', outputDir: 'must-not-be-created' };
await assert.rejects(qualifyRerankerElectron(config), /retained packageBundlePath/);
await assert.rejects(buildRerankerEvaluationPack({}), /requires qualificationPath/);
await assert.rejects(buildRerankerEvaluationPack({ qualificationPath: 'missing', conversionConfigPath: 'missing',
  licensePath: 'missing', applicationPath: 'missing', outputDir: 'must-not-be-created', authorityId: 'test' }), /explicit fail-closed/);
console.log('reranker-evaluation-preflight.test: ok (no hardware launched)');
