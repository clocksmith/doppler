import assert from 'node:assert/strict';
import { buildRerankFairnessAudit, buildSummary } from '../../tools/compare-rerankers.js';

const hardware = { vendor: 'amd', architecture: 'rdna-3', device: 'test-device', description: 'test adapter' };
const env = { runtime: 'browser', device: 'webgpu', browserUserAgent: 'same browser build', browserPlatform: 'Linux',
  dtypeFallbackUsed: false, executionProviderFallbackUsed: false, ortProxyFallbackUsed: false,
  executionProviderMode: 'webgpu-only' };
const metrics = { warmupRuns: 1, timedRuns: 3, validRuns: 3, invalidRuns: 0,
  semanticPassed: true, topDocumentIndex: 0, nonFiniteScores: 0 };
const dopplerBench = { env, deviceInfo: { adapterInfo: hardware }, cacheMode: 'warm', loadMode: 'http', metrics };
const tjsBench = { ...dopplerBench, deviceInfo: hardware, warmupRuns: 1, timedRuns: 3 };
const inputs = { profile: { compareLane: 'performance_comparable', releaseClaimable: true },
  dopplerSource: 'quickstart-registry', dopplerBench, dopplerVerify: dopplerBench, tjsBench };
const audit = value => buildRerankFairnessAudit({ ...value,
  summary: buildSummary(value.dopplerBench, value.dopplerVerify, value.tjsBench, 0) });
assert.equal(audit(inputs).claimGrade, true);
for (const change of [
  value => { value.dopplerBench.env.runtime = 'node'; },
  value => { value.tjsBench.env.browserUserAgent = 'different browser'; },
  value => { value.tjsBench.env.device = 'wasm'; },
  value => { value.tjsBench.deviceInfo.device = 'different GPU'; },
  value => { value.tjsBench.cacheMode = 'cold'; },
  value => { value.tjsBench.timedRuns = 2; },
  value => { value.dopplerBench.metrics.invalidRuns = 1; },
  value => { value.dopplerBench.metrics.validRuns = 2; },
  value => { value.tjsBench.env.executionProviderFallbackUsed = true; },
  value => { value.tjsBench.env.dtypeFallbackUsed = true; },
  value => { delete value.tjsBench.env.ortProxyFallbackUsed; },
  value => { value.dopplerBench.metrics.semanticPassed = false; },
  value => { value.dopplerBench.metrics.topDocumentIndex = 1; },
]) {
  const changed = JSON.parse(JSON.stringify(inputs)); change(changed);
  const result = audit(changed);
  assert.equal(result.claimGrade, false); assert.equal(result.releaseClaimable, false);
  assert(result.invalidReasons.length > 0);
}
assert.equal(buildRerankFairnessAudit({ profile: inputs.profile, dopplerSource: inputs.dopplerSource,
  summary: { correctnessOk: true } }).claimGrade, false, 'Missing observed scope cannot pass.');
console.log('reranker-observed-fairness.test: ok (synthetic scope rejection, not performance evidence)');
