import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../../demo/index.html', import.meta.url), 'utf8');
const xraySource = readFileSync(new URL('../../demo/ui/xray/index.js', import.meta.url), 'utf8');
const reportSource = readFileSync(new URL('../../demo/report.js', import.meta.url), 'utf8');
const settingsSource = readFileSync(new URL('../../demo/settings.js', import.meta.url), 'utf8');
const inputSource = readFileSync(new URL('../../demo/input.js', import.meta.url), 'utf8');
const modelsSource = readFileSync(new URL('../../demo/models.js', import.meta.url), 'utf8');
const tokenPressRendererSource = readFileSync(
  new URL('../../demo/ui/token-press/renderer.js', import.meta.url),
  'utf8'
);
const appStyles = readFileSync(new URL('../../demo/ui/styles/app.css', import.meta.url), 'utf8');
const tokenPressStyles = readFileSync(
  new URL('../../demo/ui/token-press/styles.css', import.meta.url),
  'utf8'
);
const componentStyles = readFileSync(new URL('../../demo/styles/rd-components.css', import.meta.url), 'utf8');

assert.match(html, /Local WebGPU inference, inspected in your browser\./);
assert.match(html, /href="https:\/\/github\.com\/clocksmith\/doppler"/);
assert.match(html, /href="https:\/\/www\.npmjs\.com\/package\/doppler-gpu"/);
assert.match(html, /Runs locally in this browser\. The first use downloads the selected model\./);
assert.match(html, /id="model-select"/);
assert.match(html, /id="model-select-action"/);
assert.match(html, /id="model-browser" class="model-browser"/);
assert.match(modelsSource, /function renderModelSelector\(\)/);
assert.match(modelsSource, /getModelActionLabel\(status\)/);

assert.equal((html.match(/id="xray-toggle-all"/g) ?? []).length, 1);
assert.doesNotMatch(html, /id="xray-toggle-(?:decode|kv|kernel|gpu|exec|mem|batch)"/);
assert.match(html, /<span class="chat-toggle-label">X-Ray<\/span>\s*<input id="xray-toggle-all" type="checkbox" checked>/);
assert.match(html, /<span class="chat-toggle-label">Token perplexity<\/span>\s*<input id="set-token-press" type="checkbox" checked>/);
assert.doesNotMatch(html, /all internals|choices \+ confidence/);
assert.match(html, /class="chat-toolbar"/);
assert.match(html, /id="chat-controls" class="chat-controls"/);
assert.match(html, /X-Ray on · Perplexity on/);
assert.match(html, /id="xray-shell" class="xray-shell" hidden/);
assert.match(html, /id="xray-summary-state"[^>]*>Enabled · 7 panels<\/span>/);
assert.ok(html.indexOf('id="chat-heading"') < html.indexOf('class="chat-toolbar"'));
assert.ok(html.indexOf('id="xray-toggle-all"') < html.indexOf('id="output-toks"'));
assert.ok(html.indexOf('id="settings-toggle"') < html.indexOf('id="output-toks"'));
assert.ok(html.indexOf('id="xray-toggle-all"') < html.indexOf('id="settings-toggle"'));
assert.ok(html.indexOf('id="settings-toggle"') < html.indexOf('id="output-area"'));
assert.match(xraySource, /\$\('xray-toggle-all'\)\?\.checked === true/);
assert.match(componentStyles, /input\[type="checkbox"\]:checked::before\s*\{\s*transform: scale\(1\)/);
assert.match(appStyles, /\.chat-toggle:has\(input:checked\)/);
assert.match(appStyles, /grid-auto-rows: 1fr/);
assert.match(appStyles, /\.model-card\s*\{[\s\S]*?height: 100%/);
assert.match(appStyles, /\.input-row textarea\s*\{[\s\S]*?overflow-y: auto/);
assert.match(appStyles, /\.chat-surface\s*\{[\s\S]*?overflow: auto/);
assert.match(tokenPressStyles, /\.tp-alternatives\s*\{[\s\S]*?position: fixed/);
assert.match(tokenPressRendererSource, /function positionAlternativesTooltip\(span\)/);
assert.match(tokenPressRendererSource, /viewportWidth - tooltipRect\.width/);
assert.match(settingsSource, /doppler\.demo\.token-perplexity-enabled/);
assert.match(xraySource, /doppler\.demo\.xray-enabled/);

assert.match(html, /id="set-max-tokens"[^>]*value="1024"/);
assert.match(settingsSource, /DEMO_DEFAULT_MAX_TOKENS = 1024/);
assert.match(html, /id="shuffle-btn"[^>]*>[\s\S]*Shuffle<\/button>/);
assert.match(html, /id="image-drop"[^>]*>Attach image<\/button>/);
assert.match(html, /id="run-btn"[^>]*>[\s\S]*&#x25B8;[\s\S]*Send<\/button>/);
assert.match(html, /id="status-text">Select model<\/span>/);
assert.match(inputSource, /addEventListener\('input', \(\) => syncSendButton\(\)\)/);
assert.ok(html.indexOf('id="shuffle-btn"') < html.indexOf('id="image-drop"'));
assert.ok(html.indexOf('id="image-drop"') < html.indexOf('id="run-btn"'));

assert.doesNotMatch(html, /id="history-toggle"/);
assert.doesNotMatch(html, /id="history-limit"/);
assert.doesNotMatch(html, /id="history-status"/);
assert.match(html, /id="clear-history-btn"[^>]*>Clear chat<\/button>/);
assert.match(html, /id="chat-thread"/);
assert.ok(html.indexOf('id="output-area"') < html.indexOf('id="input-area"'));

assert.ok(html.indexOf('id="export-btn"') < html.indexOf('id="import-btn"'));
assert.ok(html.indexOf('id="import-btn"') < html.indexOf('id="precision-replay-toggle"'));
assert.ok(html.indexOf('id="precision-replay-toggle"') < html.indexOf('id="precision-replay-panel"'));
assert.match(reportSource, /importReportData/);
assert.match(reportSource, /import-file/);

console.log('demo-ui-controls.test: ok');
