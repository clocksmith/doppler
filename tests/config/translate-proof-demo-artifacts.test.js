import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const demoSource = await readFile(new URL('../../demo/demo-core.js', import.meta.url), 'utf8');
const demoHtml = await readFile(new URL('../../demo/index.html', import.meta.url), 'utf8');
const demoState = await readFile(new URL('../../demo/ui/state.js', import.meta.url), 'utf8');

assert.match(demoSource, /const TRANSLATE_COMPARE_DEFAULT_MAX_TOKENS = 192;/);
assert.match(demoSource, /const TRANSLATE_COMPARE_ARTIFACT_KIND = 'doppler\.translate\.compare\/v1';/);
assert.match(demoSource, /const TRANSLATE_COMPARE_CONFIG_VERSION = 2;/);

assert.equal((demoSource.match(/bucket: 'easy'/g) || []).length, 3, 'smoke panel should include 3 easy samples');
assert.equal((demoSource.match(/bucket: 'nuanced'/g) || []).length, 3, 'smoke panel should include 3 nuanced samples');
assert.equal((demoSource.match(/bucket: 'domain'/g) || []).length, 2, 'smoke panel should include 2 domain samples');
assert.equal((demoSource.match(/bucket: 'edge'/g) || []).length, 2, 'smoke panel should include 2 edge samples');

assert.match(demoSource, /data-compare-history-export=/);
assert.match(demoSource, /function buildTranslateCompareArtifact\(/);
assert.match(demoSource, /function renderTranslateCompareReceipts\(/);
assert.match(demoSource, /function renderTranslateCompareSmokePanel\(/);

assert.match(demoState, /compareHistoryFilter: 'all',/);
assert.match(demoState, /activeCompareSmokeSampleId: null,/);
assert.match(demoState, /lastCompareArtifact: null,/);
assert.doesNotMatch(demoState, /tjsDtype: 'fp16'/);

assert.match(demoHtml, /id="translate-compare-export-btn"/);
assert.match(demoHtml, /id="translate-compare-export-latest-btn"/);
assert.match(demoHtml, /id="translate-compare-receipts"/);
assert.match(demoHtml, /id="translate-smoke-panel"/);
assert.match(demoHtml, /id="translate-smoke-grid"/);
assert.match(demoHtml, /data-compare-history-filter="all"/);
assert.match(demoHtml, /data-compare-history-filter="same-model"/);
assert.match(demoHtml, /data-compare-history-filter="same-engine"/);
assert.match(demoHtml, /data-compare-history-filter="proof"/);

console.log('translate-proof-demo-artifacts.test: ok');
