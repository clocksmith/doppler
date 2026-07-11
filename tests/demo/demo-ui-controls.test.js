import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../../demo/index.html', import.meta.url), 'utf8');
const xraySource = readFileSync(new URL('../../demo/ui/xray/index.js', import.meta.url), 'utf8');
const reportSource = readFileSync(new URL('../../demo/report.js', import.meta.url), 'utf8');
const settingsSource = readFileSync(new URL('../../demo/settings.js', import.meta.url), 'utf8');

assert.match(html, /Local WebGPU inference, inspected in your browser\./);
assert.match(html, /href="https:\/\/github\.com\/clocksmith\/doppler"/);
assert.match(html, /href="https:\/\/www\.npmjs\.com\/package\/doppler-gpu"/);
assert.match(html, /Choose a model to run in your browser\. Note the download size\./);

assert.equal((html.match(/id="xray-toggle-all"/g) ?? []).length, 1);
assert.doesNotMatch(html, /id="xray-toggle-(?:decode|kv|kernel|gpu|exec|mem|batch)"/);
assert.match(html, /id="set-token-press"[\s\S]*?<strong>Token logits<\/strong>/);
assert.match(html, /class="chat-toolbar"/);
assert.ok(html.indexOf('id="xray-toggle-all"') < html.indexOf('id="output-toks"'));
assert.ok(html.indexOf('id="settings-toggle"') < html.indexOf('id="output-toks"'));
assert.ok(html.indexOf('id="xray-toggle-all"') < html.indexOf('id="settings-toggle"'));
assert.ok(html.indexOf('id="settings-toggle"') < html.indexOf('id="output-area"'));
assert.match(xraySource, /\$\('xray-toggle-all'\)\?\.checked === true/);

assert.match(html, /id="set-max-tokens"[^>]*value="1024"/);
assert.match(settingsSource, /DEMO_DEFAULT_MAX_TOKENS = 1024/);
assert.match(html, /id="shuffle-btn"[^>]*>[\s\S]*Shuffle<\/button>/);
assert.match(html, /id="run-btn"[^>]*>[\s\S]*&#x25B8;[\s\S]*Run<\/button>/);

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
