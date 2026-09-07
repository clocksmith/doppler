import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { auditRerankerReferencePair } from '../../tools/compare-reranker-reference-runs.js';
import { buildRerankerReferenceSchedule } from '../../tools/reranker-reference-schedule.js';

const bytes = await fs.readFile(new URL('../../reports/retention-experiment/20260907/inputs/heldout-recovery-reference.json', import.meta.url));
const reference = JSON.parse(bytes);
const digest = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
const references = [{ digest, reference }];
const sampling = { warmupRuns: 1, timedRuns: 3 };
const runSchedule = buildRerankerReferenceSchedule(sampling);
const base = {
  passed: true, cleanup: [], sampling, runSchedule,
  config: { repeatRuns: 1, references: [{ digest }], launchArgs: [], sampleIntervalMs: 25, dtype: 'q4' },
  hardware: { vendor: 'test', architecture: 'test', device: 'test', description: 'test', isFallbackAdapter: false },
  browser: { product: 'test', revision: 'test', userAgent: 'test', jsVersion: 'test' },
  qualifierDigest: digest, scheduleDigest: digest, memorySamplerDigest: digest,
  startup: { scope: 'test', modelReadyMs: 10, firstResultMs: 12 }, peakRendererRssBytes: 100,
  phases: [{ repeat: 0, observation: { modelLoadMs: 5, scoringConfig: reference.scoringConfig,
    executionProviderMode: 'webgpu-only', requestedDtype: 'q4', effectiveDtype: 'q4',
    fallbackUsed: false, executionProviderFallbackUsed: false, ortProxyFallbackUsed: false,
    runs: runSchedule.map(sample => ({ ...sample, referenceIndex: 0, durationMs: 2,
      ...reference.input, receipt: { evidence: reference.input }, scores: reference.outputs })) } }],
};
const pair = () => ({
  doppler: { ...structuredClone(base), schema: 'doppler.installed-browser-reranker-qualification-result/v1' },
  tjs: { ...structuredClone(base), schema: 'doppler.transformersjs-reranker-qualification-result/v1' },
});
const evaluate = ({ doppler, tjs }) => auditRerankerReferencePair(doppler, tjs, references);
assert.equal(evaluate(pair()).timingComparable, true);
assert.equal(evaluate(pair()).claimAllowed, false, 'One pair does not establish a repeated performance or lifecycle claim.');
for (const mutate of [
  p => { p.tjs.phases[0].observation.runs[0].scores[0].tokenIds[0]++; },
  p => { p.doppler.phases[0].observation.runs[0].scores[0].trueLogit += 100; },
  p => { p.tjs.phases[0].observation.runs[0].query = 'different input'; },
  p => { p.doppler.phases[0].observation.runs[0].receipt.evidence.query = 'different input'; },
  p => { p.tjs.phases[0].observation.runs.pop(); },
  p => { p.tjs.phases[0].observation.effectiveDtype = 'fp32'; },
  p => { p.tjs.phases[0].observation.executionProviderFallbackUsed = true; },
  p => { p.tjs.browser.revision = 'different'; },
  p => { delete p.tjs.memorySamplerDigest; delete p.doppler.memorySamplerDigest; },
  p => { p.tjs.hardware.device = 'different'; },
  p => { p.tjs.phases[0].observation.runs[0].durationMs = NaN; },
  p => { p.doppler.config.repeatRuns = 2; },
  p => { p.tjs.startup.scope = 'different'; },
]) {
  const altered = pair(); mutate(altered);
  assert.equal(evaluate(altered).timingComparable, false);
}
const qualified = pair();
for (const report of Object.values(qualified)) {
  report.sampling = null; report.runSchedule = buildRerankerReferenceSchedule(null);
  report.phases[0].observation.runs = [{ ...report.phases[0].observation.runs[0], phase: 'reference' }];
}
assert.equal(evaluate(qualified).qualityPassed, true);
assert.equal(evaluate(qualified).timingComparable, false, 'Reference probes cannot be relabeled as calibration.');
console.log('reranker-reference-pair-audit.test: ok (synthetic rejection, not hardware evidence)');
