import test from 'node:test';
import assert from 'node:assert/strict';
import { assess, compareResults, describeBrowser, readResult } from '../public/results.js';

const TOKENS = [5, 6, 7];
const protocol = {
  schema: 'doppler.observatory.protocol/v1', id: 'device-benchmark', version: '0.2.0',
  workload: { modelId: 'm', prompt: 'p', observationPolicy: 'demo/always-on', warmRuns: 2, generation: { temperature: 0 } },
  reference: { generatedTokenIds: TOKENS },
  eligibility: { requireVisibleTab: true, requireIdenticalTokensAcrossRuns: true, requireReferenceMatchWhenPinned: true, requirePlainPathCheck: true, maxPlainPathDeviation: 0.15 },
};
const run = (wallMs, rate, tokens = TOKENS) => ({ generatedTokenIds: tokens, wallMs, timeToFirstTokenMs: 40, decodeTokensPerSecond: rate, sampling: { temperature: 0 } });
const result = ({ rate = 30, vendor = 'apple', version = '0.2.0', tokens = TOKENS, ...rest } = {}) => ({
  schema: 'doppler.observatory.device-benchmark/v1', protocol: { id: 'device-benchmark', version }, startedAt: '2026-11-10T12:00:00.000Z',
  failure: null, tabHiddenDuringRun: false, policy: { performanceRepresentative: true, modifiesExecution: false },
  device: { userAgent: 'Mozilla/5.0 (Linux; Android 16; Pixel 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36', adapter: { vendor, architecture: 'arch' } },
  runtime: { version: '0.7.0', release: { sourceCommit: 'e56cf5d668a561cfafe579e98ed3a7c2a91ee4bf' } },
  load: { wasStored: true, wallMs: 2000 }, run: { id: 'run-0123456789abcdef' },
  cold: run(900, rate, tokens), warm: [run(500, rate, tokens), run(520, rate, tokens)], plainPath: [run(505, rate, tokens), run(515, rate, tokens)],
  // What the file claims about itself must not matter.
  verdict: { eligible: true, state: 'passed' },
  ...rest,
});

test('a result file is checked before it is read', () => {
  assert.equal(readResult(result()).schema, 'doppler.observatory.device-benchmark/v1');
  assert.throws(() => readResult({ name: 'x' }), /does not say what it holds/);
  assert.throws(() => readResult({ schema: 'doppler.observatory.run/v1' }), /its format is "doppler\.observatory\.run\/v1"/);
  assert.throws(() => readResult(result({ warm: [{ generatedTokenIds: 'no' }] })), /its runs are malformed/);
  assert.throws(() => readResult(result({ startedAt: 'yesterday' })), /no valid start time/);
});

test('a result is judged here, whatever verdict the file carries', () => {
  const wrong = assess(protocol, result({ tokens: [5, 6, 9] }));
  assert.equal(wrong.state, 'not-eligible');
  assert.equal(wrong.comparable, false);
  assert.match(wrong.reasons[0], /pinned reference/);
  const fine = assess(protocol, result());
  assert.deepEqual([fine.state, fine.comparable, fine.browser, fine.adapter, fine.runtime, fine.runId],
    ['passed', true, 'Chrome 145 on Android', 'apple arch', 'Doppler 0.7.0 (e56cf5d6)', 'run-0123456789abcdef']);
});

test('a result under another protocol version is shown but not put beside the others', () => {
  const old = assess(protocol, result({ version: '0.1.0' }));
  assert.deepEqual([old.state, old.comparable, old.summary], ['other-protocol', false, null]);
  assert.match(old.reasons[0], /Recorded under device-benchmark 0\.1\.0\. This page judges device-benchmark 0\.2\.0/);
});

test('only eligible results are measured against each other, from the first one opened', () => {
  const view = compareResults(protocol, [result({ tokens: [5, 6, 9], rate: 99 }), result({ rate: 30 }), result({ rate: 45, vendor: 'qualcomm' }), result({ version: '0.1.0', rate: 500 })]);
  assert.deepEqual(view.rows.map((row) => row.relativeDecodeRate), [null, 1, 1.5, null]);
  assert.deepEqual([view.eligible, view.devices, view.runtimes, view.basisIndex], [2, 2, 1, 1]);
  const none = compareResults(protocol, [result({ failure: { stage: 'load', message: 'No adapter' }, cold: null, warm: [], plainPath: null })]);
  assert.deepEqual([none.eligible, none.basisIndex, none.rows[0].state], [0, null, 'failed']);
});

test('results from a preview runtime are counted and marked', () => {
  const preview = result({ rate: 40, runtime: { version: '0.7.0', channel: 'preview', preview: true, release: { sourceCommit: 'b9116524d9a2389fd235fa32b8312ac1b0f77d45' } } });
  const view = compareResults(protocol, [result(), preview]);
  assert.deepEqual(view.rows.map((row) => row.previewRuntime), [false, true]);
  assert.deepEqual([view.eligible, view.previews, view.runtimes], [2, 1, 2]);
  assert.match(view.rows[1].notes[0], /preview runtime/);
});

test('browsers are named from their user agent', () => {
  assert.equal(describeBrowser('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/145.0.7632.6 Safari/537.36'), 'Chrome (headless) 145 on macOS');
  assert.equal(describeBrowser('Mozilla/5.0 (iPhone; CPU iPhone OS 19_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/19.1 Mobile/15E148 Safari/604.1'), 'Safari 19 on iOS');
  assert.equal(describeBrowser('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0'), 'Edge 145 on Windows');
  assert.equal(describeBrowser('Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0'), 'Firefox 140 on Linux');
  assert.equal(describeBrowser(null), 'browser not recorded');
});
