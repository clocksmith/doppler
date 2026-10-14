import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getXrayRuntimeNoticeText } from '../../demo/ui/xray/index.js';

const demo = readFileSync(new URL('../../demo/demo.js', import.meta.url), 'utf8');
const core = readFileSync(new URL('../../demo/core.js', import.meta.url), 'utf8');
const models = readFileSync(new URL('../../demo/models.js', import.meta.url), 'utf8');

assert.match(getXrayRuntimeNoticeText({ profilingEnabled: true }), /Timings are diagnostic/);
assert.match(models, /from 'doppler-gpu\/compat'/);
assert.doesNotMatch(`${demo}\n${core}\n${models}`, /(?:\.\.\/)+src\//);

console.log('demo-surface-contract.test: ok');
