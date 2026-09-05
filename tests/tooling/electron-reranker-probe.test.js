import assert from 'node:assert/strict';
import { assertPhysicalAdapter, assertRanking } from '../../tools/probe-electron-reranker.js';

assertPhysicalAdapter({ vendor: 'amd', architecture: 'rdna-3' }, 'amd');
for (const adapter of [{ vendor: 'amd', isFallbackAdapter: true }, { vendor: 'amd', description: 'SwiftShader' }, { vendor: 'intel' }]) {
  assert.throws(() => assertPhysicalAdapter(adapter, 'amd'));
}
const policy = { documents: ['a', 'b'], acceptance: { topDocumentIndex: 0, requireFiniteScores: true } };
const report = {
  metrics: { topDocumentIndex: 0 },
  output: { ranking: [{ index: 0, document: 'a', score: 2 }, { index: 1, document: 'b', score: 1 }] },
  results: [{ passed: true }],
};
assertRanking(report, policy);
assert.throws(() => assertRanking({ ...report, results: [{ passed: false }] }, policy));
assert.throws(() => assertRanking({ ...report, metrics: { ...report.metrics, topDocumentIndex: 1 } }, policy));
assert.throws(() => assertRanking({ ...report, output: { ranking: [{ index: 0, document: 'a', score: NaN }, report.output.ranking[1]] } }, policy));
assert.throws(() => assertRanking({ ...report, output: { ranking: [report.output.ranking[0], report.output.ranking[0]] } }, policy));
assert.throws(() => assertRanking({ ...report, output: { ranking: [report.output.ranking[0], { index: 1, document: 'other', score: 1 }] } }, policy));
console.log('electron-reranker-probe.test: ok');
