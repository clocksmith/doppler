import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../../demo/index.html', import.meta.url), 'utf8');
const demo = readFileSync(new URL('../../demo/demo.js', import.meta.url), 'utf8');
const core = readFileSync(new URL('../../demo/core.js', import.meta.url), 'utf8');
const models = readFileSync(new URL('../../demo/models.js', import.meta.url), 'utf8');

assert.match(html, /src="\/demo\/demo\.js"/);
assert.doesNotMatch(html, /translate-compare-shell/);
assert.doesNotMatch(html, /id="xray-toggle-all"[^>]*checked/);
assert.doesNotMatch(html, /id="set-word-quality"[^>]*checked/);
assert.match(demo, /from '\.\/core\.js'/);
assert.match(models, /from 'doppler-gpu\/compat'/);
assert.match(core, /model\.inspect\.generate/);
assert.match(core, /demo\/always-on/);
assert.match(core, /demo\/guided-quality/);
assert.match(core, /demo\/deep-xray/);
assert.doesNotMatch(`${demo}\n${core}\n${models}`, /(?:\.\.\/)+src\//);

console.log('demo-surface-contract.test: ok');
