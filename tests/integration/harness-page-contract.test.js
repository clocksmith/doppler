import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const harnessHtml = readFileSync(new URL('../harness.html', import.meta.url), 'utf8');
const harnessReadme = readFileSync(new URL('../README.md', import.meta.url), 'utf8');

assert.match(harnessHtml, /applyRuntimeForRun/);
assert.match(
  harnessHtml,
  /await applyRuntimeForRun\(\{[\s\S]*configChain:[\s\S]*runtimeProfile,[\s\S]*runtimeConfigUrl,[\s\S]*runtimeConfig:/,
);
assert.match(harnessHtml, /mode: params\.get\('mode'\) \|\| 'verify'/);
assert.match(harnessHtml, /workload: params\.get\('workload'\) \|\| 'kernels'/);

assert.match(harnessReadme, /runtimeProfile/);
assert.match(harnessReadme, /fail-closed auto-surface exceptions/);

console.log('harness-page-contract.test: ok');
