import assert from 'node:assert/strict';
import { APP_SHELL } from '../../demo/generated-shell-manifest.js';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../../demo/index.html', import.meta.url), 'utf8');

for (const match of html.matchAll(/<link rel="stylesheet" href="([^"]+)"/g)) {
  assert.ok(APP_SHELL.includes(match[1]), `Stylesheet missing from offline shell: ${match[1]}`);
}

console.log('demo-ui-controls.test: ok');
