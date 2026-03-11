import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const demoHtml = readFileSync(new URL('../../demo/index.html', import.meta.url), 'utf8');
const demoSource = readFileSync(new URL('../../demo/demo.js', import.meta.url), 'utf8');

assert.match(demoHtml, /id="translate-compare-shell"/);
assert.match(demoHtml, /id="translate-compare-preset"/);
assert.match(demoHtml, /id="translate-compare-run-btn"/);
assert.match(demoHtml, /id="translate-history-list"/);
assert.match(demoHtml, /id="translate-left-engine"/);
assert.match(demoHtml, /id="translate-right-engine"/);

assert.match(demoSource, /async function handleTranslateCompareRun\(\)/);
assert.match(demoSource, /async function applyTranslateComparePreset\(presetId, options = \{\}\)/);
assert.match(demoSource, /function syncTranslateCompareUI\(\)/);
assert.match(demoSource, /loadTranslateCompareProfiles\(\)/);
assert.match(demoSource, /async function loadTranslateCompareEvidence\(\)/);

console.log('translate-compare-demo-contract.test: ok');
