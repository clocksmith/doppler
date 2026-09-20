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
const tokenInspectorStyles = readFileSync(
  new URL('../../demo/ui/token-inspector/styles.css', import.meta.url),
  'utf8'
);
const appStyles = readFileSync(new URL('../../demo/ui/styles/app.css', import.meta.url), 'utf8');
const xrayStyles = readFileSync(new URL('../../demo/ui/xray/styles.css', import.meta.url), 'utf8');

assert.match(html, /Run a verified local model with WebGPU and inspect the evidence behind every generation\./);
assert.match(html, /<h1 class="app-brand-name">Doppler<\/h1>/);
assert.match(html, /id="inspection-workspace"/);
assert.match(html, /src="\/demo\/demo\.js"/);
assert.match(html, /id="model-select"/);
assert.match(html, /id="model-select-action"/);
assert.match(modelsSource, /from 'doppler-gpu\/compat'/);

assert.equal((html.match(/id="xray-toggle-all"/g) ?? []).length, 1);
assert.match(html, /<span class="chat-toggle-label">X-Ray<\/span>\s*<input id="xray-toggle-all" type="checkbox">/);
assert.match(html, /<span class="chat-toggle-label">Perplexity<\/span>\s*<input id="set-word-quality" type="checkbox">/);
assert.match(html, /chat-controls-summary-state">Standard/);
assert.match(html, /Enabled · 5 evidence panels/);
assert.doesNotMatch(html, /capture-transcript|export-transcript|set-token-press/);
assert.match(xraySource, /GPU timestamp queries/);
assert.match(xraySource, /canonical fingerprint matches/);
assert.match(xraySource, /localStorage\.getItem\(XRAY_STORAGE_KEY\) === 'true'/);
assert.match(wordQualityStyles, /\.word-quality/);
assert.doesNotMatch(wordQualityStyles, /\.tp-token|\.tp-alternatives/);
assert.match(tokenInspectorStyles, /\.token-chip\s*\{/);
assert.match(tokenInspectorStyles, /text-decoration-color:\s*color-mix/);
assert.doesNotMatch(tokenInspectorStyles, /\.token-chip\s*\{[^}]*border:\s*1px/s);
assert.match(appStyles, /--doppler-canvas:\s*#ffffff/);
assert.match(appStyles, /--doppler-surface:\s*#ffffff/);
assert.match(appStyles, /--doppler-soft-surface:\s*#ffffff/);
assert.deepEqual(
  appStyles.match(/background(?:-color)?:\s*var\(--doppler-(?:blue|purple|red)\)/g),
  ['background: var(--doppler-blue)']
);
assert.match(appStyles, /\.status-dot\.is-ready\s*\{\s*background:\s*var\(--doppler-blue\)/);
assert.doesNotMatch(xrayStyles, /--xray-(?:green|orange|magenta)/);
assert.match(xrayStyles, /--xray-surface:\s*var\(--bg,\s*#fff\)/);
assert.match(settingsSource, /doppler\.demo\.word-quality-enabled/);

assert.match(html, /<select id="set-max-tokens"[^>]*>[\s\S]*<option value="256" selected>/);
assert.match(settingsSource, /DEMO_DEFAULT_MAX_TOKENS = 256/);
assert.match(html, /id="shuffle-btn"[^>]*>[\s\S]*Example<\/button>/);
// Inspection accepts text and returns completed timing; don't offer disconnected controls.
assert.doesNotMatch(html, /id="image-drop"|id="set-live-toks"/);
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
