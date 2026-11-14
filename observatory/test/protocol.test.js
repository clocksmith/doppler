import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { executionDifference, firstDifference, judgeRun, median, spread, validateProtocol } from '../public/protocol.js';

const TOKENS = [5, 6, 7, 8];
const protocol = (overrides = {}) => ({
  schema: 'doppler.observatory.protocol/v1',
  id: 'device-benchmark',
  version: '0.1.0',
  workload: { modelId: 'm', manifestHash: 'sha256:aa', prompt: 'p', observationPolicy: 'demo/always-on', warmRuns: 2, generation: { temperature: 0, repetitionPenalty: 1 } },
  reference: { generatedTokenIds: TOKENS },
  eligibility: {
    requireVisibleTab: true,
    requireProtocolArtifact: true,
    requirePolicyPerformanceRepresentative: true,
    requirePolicyDoesNotModifyExecution: true,
    requireProtocolSampling: true,
    requireIdenticalTokensAcrossRuns: true,
    requireReferenceMatchWhenPinned: true,
    requirePlainPathCheck: true,
    requireSameExecutionAsPlainPath: true,
    maxPlainPathDeviation: 0.15,
    maxRepeatSpread: 0.25,
  },
  ...overrides,
});
const SETTINGS = { decodeMode: 'single_token', batchSize: 1, disableCommandBatching: false };
const SAMPLING = { temperature: 0, repetitionPenalty: 1, topK: 1 };
const run = (wallMs, tokens = TOKENS, execution = SETTINGS, sampling = SAMPLING) => ({ generatedTokenIds: tokens, wallMs, timeToFirstTokenMs: 40, decodeTokensPerSecond: 30, execution, sampling });
const evidence = (overrides = {}) => ({
  failure: null,
  tabHiddenDuringRun: false,
  fingerprint: { identity: { artifact: { modelId: 'm', manifestHash: 'sha256:aa' } } },
  policy: { id: 'demo/always-on', performanceRepresentative: true, modifiesExecution: false },
  cold: run(900),
  warm: [run(500), run(520)],
  plainPath: [run(505), run(515)],
  ...overrides,
});

test('the checked-in protocol is valid', async () => {
  validateProtocol(JSON.parse(await readFile(new URL('../config/protocols/device-benchmark.json', import.meta.url), 'utf8')));
});

test('a clean run passes and reports medians', () => {
  const verdict = judgeRun(protocol(), evidence());
  assert.equal(verdict.state, 'passed');
  assert.deepEqual(verdict.reasons, []);
  assert.equal(verdict.summary.warmWallMs, 510);
  assert.equal(verdict.summary.plainWallMs, 510);
  assert.equal(verdict.summary.referenceMatch, true);
});

test('a failed run reports where it stopped and no numbers', () => {
  const verdict = judgeRun(protocol(), evidence({ failure: { stage: 'load', message: 'No adapter' } }));
  assert.equal(verdict.state, 'failed');
  assert.match(verdict.reasons[0], /load.*No adapter/);
  assert.equal(verdict.summary, null);
});

test('an answer that differs from the reference names the first differing token', () => {
  const wrong = [5, 6, 9, 8];
  const verdict = judgeRun(protocol(), evidence({ cold: run(900, wrong), warm: [run(500, wrong), run(520, wrong)], plainPath: [run(505, wrong), run(515, wrong)] }));
  assert.equal(verdict.state, 'not-eligible');
  assert.equal(verdict.reasons.length, 1);
  assert.match(verdict.reasons[0], /token 3/);
});

test('runs that disagree with each other are not eligible', () => {
  const verdict = judgeRun(protocol(), evidence({ warm: [run(500), run(520, [5, 6, 7, 1])] }));
  assert.ok(verdict.reasons.some((reason) => /different tokens/.test(reason)));
});

test('an unpinned reference is a note, not a pass on correctness', () => {
  const verdict = judgeRun(protocol({ reference: { generatedTokenIds: null } }), evidence());
  assert.equal(verdict.state, 'passed');
  assert.equal(verdict.summary.referenceMatch, null);
  assert.match(verdict.notes[0], /No reference answer is pinned/);
});

test('a tier that changes execution or is not performance-representative is refused', () => {
  const verdict = judgeRun(protocol(), evidence({ policy: { performanceRepresentative: false, modifiesExecution: true } }));
  assert.equal(verdict.reasons.length, 2);
});

test('timing that drifts from generation without inspection is refused in both directions', () => {
  const slower = judgeRun(protocol(), evidence({ warm: [run(700), run(700)], plainPath: [run(500), run(500)] }));
  assert.match(slower.reasons[0], /40\.0% longer/);
  const faster = judgeRun(protocol(), evidence({ warm: [run(300), run(300)], plainPath: [run(500), run(500)] }));
  assert.match(faster.reasons[0], /40\.0% less time/);
});

test('a missing cross-check is refused when the protocol requires it and noted when it does not', () => {
  assert.equal(judgeRun(protocol(), evidence({ plainPath: null })).state, 'not-eligible');
  const relaxed = protocol();
  relaxed.eligibility.requirePlainPathCheck = false;
  const verdict = judgeRun(relaxed, evidence({ plainPath: null }));
  assert.equal(verdict.state, 'passed');
  assert.match(verdict.notes[0], /not cross-checked/);
});

test('a runtime that sampled differently from the protocol is refused', () => {
  const penalized = { ...SAMPLING, repetitionPenalty: 1.1 };
  const verdict = judgeRun(protocol(), evidence({ warm: [run(500), run(520, TOKENS, SETTINGS, penalized)] }));
  assert.deepEqual(verdict.reasons, ['The runtime did not use the protocol\'s sampling settings (repetitionPenalty).']);
  assert.equal(judgeRun(protocol(), evidence({ cold: run(900, TOKENS, SETTINGS, null) })).state, 'not-eligible', 'sampling that was not reported is not assumed');
});

test('a result made on a preview runtime says so and is still judged', () => {
  const verdict = judgeRun(protocol(), evidence({ runtime: { version: '0.7.0', channel: 'preview', preview: true } }));
  assert.equal(verdict.state, 'passed');
  assert.match(verdict.notes[0], /preview runtime, a build that is not qualified/);
  assert.deepEqual(judgeRun(protocol(), evidence({ runtime: { version: '0.6.36', channel: 'hosted', preview: false } })).notes, []);
});

test('other model files than the protocol names are refused', () => {
  const verdict = judgeRun(protocol(), evidence({ fingerprint: { identity: { artifact: { modelId: 'm', manifestHash: 'sha256:bb' } } } }));
  assert.deepEqual(verdict.reasons, ['The model files that ran are not the ones this protocol names.']);
  assert.equal(judgeRun(protocol(), evidence({ fingerprint: null })).state, 'not-eligible');
});

test('a hidden tab or too few runs is not eligible', () => {
  assert.ok(judgeRun(protocol(), evidence({ tabHiddenDuringRun: true })).reasons.some((reason) => /hidden/.test(reason)));
  assert.ok(judgeRun(protocol(), evidence({ warm: [run(500)], plainPath: [run(500)] })).reasons.some((reason) => /2 warm runs/.test(reason)));
});

test('a benchmark tier that batches differently from plain generation is refused', () => {
  const unbatched = { ...SETTINGS, disableCommandBatching: true };
  const verdict = judgeRun(protocol(), evidence({ plainPath: [run(505, TOKENS, unbatched), run(515, TOKENS, unbatched)] }));
  assert.equal(verdict.state, 'not-eligible');
  assert.match(verdict.reasons[0], /disableCommandBatching: false and true/);
  const unreported = judgeRun(protocol(), evidence({ plainPath: [run(505, TOKENS, null), run(515, TOKENS, null)] }));
  assert.match(unreported.reasons[0], /not reported/);
});

test('repeat runs that vary too much are refused as unstable', () => {
  const verdict = judgeRun(protocol(), evidence({ warm: [run(400), run(620)], plainPath: [run(505), run(515)] }));
  assert.ok(verdict.reasons.some((reason) => /varied by 55\.0%/.test(reason)));
});

test('one stray run is ignored when the protocol trims, steady drift is not', () => {
  const trimmed = protocol();
  trimmed.workload.warmRuns = 5;
  trimmed.eligibility.repeatSpreadTrim = 1;
  const five = (values) => evidence({ warm: values.map((ms) => run(ms)), plainPath: values.map((ms) => run(ms)) });
  const stray = judgeRun(trimmed, five([1014, 1326, 1354, 1501, 1275]));
  assert.equal(stray.state, 'passed');
  assert.equal(stray.summary.warmFastestMs, 1014);
  assert.equal(stray.summary.warmSlowestMs, 1501);
  const drift = judgeRun(trimmed, five([1000, 1150, 1300, 1450, 1600]));
  assert.ok(drift.reasons.some((reason) => /even without the fastest and slowest/.test(reason)));
  trimmed.workload.warmRuns = 3;
  assert.throws(() => validateProtocol(trimmed), /fewer than two repeat runs/);
});

test('helpers', () => {
  assert.equal(spread([100, 125]), 0.25);
  assert.equal(spread([100]), null);
  assert.equal(executionDifference(SETTINGS, { ...SETTINGS }), null);
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(median([null, undefined]), null);
  assert.equal(firstDifference([1, 2], [1, 2]), -1);
  assert.equal(firstDifference([1, 2], [1, 2, 3]), 2);
});
