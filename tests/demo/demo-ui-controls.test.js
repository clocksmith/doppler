import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../../demo/index.html', import.meta.url), 'utf8');
const xraySource = readFileSync(new URL('../../demo/ui/xray/index.js', import.meta.url), 'utf8');
const reportSource = readFileSync(new URL('../../demo/report.js', import.meta.url), 'utf8');
const settingsSource = readFileSync(new URL('../../demo/settings.js', import.meta.url), 'utf8');
const inputSource = readFileSync(new URL('../../demo/input.js', import.meta.url), 'utf8');
const modelsSource = readFileSync(new URL('../../demo/models.js', import.meta.url), 'utf8');
const wordQualityStyles = readFileSync(
  new URL('../../demo/ui/word-quality/styles.css', import.meta.url),
  'utf8'
);

assert.match(html, /Local AI you can inspect\./);
assert.match(html, /Runs in this browser/);
assert.match(html, /id="inspection-workspace"/);
assert.match(html, /src="\/demo\/demo\.js"/);
assert.match(html, /id="model-select"/);
assert.match(html, /id="model-select-action"/);
assert.match(modelsSource, /from 'doppler-gpu\/compat'/);

assert.equal((html.match(/id="xray-toggle-all"/g) ?? []).length, 1);
assert.match(html, /<span class="chat-toggle-label">X-Ray<\/span>\s*<input id="xray-toggle-all" type="checkbox">/);
assert.match(html, /<span class="chat-toggle-label">Word quality<\/span>\s*<input id="set-word-quality" type="checkbox">/);
assert.match(html, /X-Ray off · Word quality off/);
assert.match(html, /Enabled · 5 evidence panels/);
assert.doesNotMatch(html, /capture-transcript|export-transcript|set-token-press/);
assert.match(xraySource, /GPU timestamp queries/);
assert.match(xraySource, /canonical fingerprint matches/);
assert.match(wordQualityStyles, /\.word-quality/);
assert.doesNotMatch(wordQualityStyles, /\.tp-token|\.tp-alternatives/);
assert.match(settingsSource, /doppler\.demo\.word-quality-enabled/);

assert.match(html, /id="set-max-tokens"[^>]*value="1024"/);
assert.match(settingsSource, /DEMO_DEFAULT_MAX_TOKENS = 1024/);
assert.match(html, /id="shuffle-btn"[^>]*>[\s\S]*Example<\/button>/);
assert.match(html, /id="image-drop"[^>]*>Image<\/button>/);
assert.match(html, /id="run-btn"[^>]*>[\s\S]*Send<\/button>/);
assert.match(inputSource, /state\.model/);

assert.ok(html.indexOf('id="export-btn"') < html.indexOf('id="import-btn"'));
assert.ok(html.indexOf('id="import-btn"') < html.indexOf('id="precision-replay-toggle"'));
assert.match(html, /Export receipt/);
assert.match(html, /Open receipt/);
assert.match(reportSource, /observationPolicy/);
assert.match(reportSource, /comparisonFingerprint/);
assert.doesNotMatch(reportSource, /lastReferenceTranscript|setTranscriptExportEnabled/);

console.log('demo-ui-controls.test: ok');
