import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const html = await fs.readFile(new URL('../../demo/index.html', import.meta.url), 'utf8');

assert.match(html, /data-mode="kernel-paths"/);
assert.match(html, /id="kernel-builder-model-select"/);
assert.match(html, /id="kernel-builder-status"/);
assert.match(html, /id="kernel-builder-proposal-json"/);
assert.match(html, /id="kernel-builder-overlay"/);

console.log('demo-kernel-path-builder-contract.test: ok');
