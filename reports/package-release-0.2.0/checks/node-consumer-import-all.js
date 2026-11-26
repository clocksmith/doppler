import assert from 'node:assert/strict';
import fs from 'node:fs';
const pkg = JSON.parse(fs.readFileSync('node_modules/dopplerx/package.json', 'utf8'));
const results = {};
for (const key of Object.keys(pkg.exports)) {
  const spec = key === '.' ? 'dopplerx' : 'dopplerx' + key.slice(1);
  const mod = await import(spec);
  results[spec] = Object.keys(mod).length;
  assert.ok(results[spec] > 0, spec + ' has no exports');
}
const root = await import('dopplerx');
assert.equal(root.DOPPLER_VERSION, '0.2.0');
const { createDopplerProvider } = await import('dopplerx/provider');
assert.equal(typeof createDopplerProvider, 'function');
console.log(JSON.stringify({ version: root.DOPPLER_VERSION, entryPoints: Object.keys(results).length, exportsPerEntry: results }));
// Node with no WebGPU provider installed: a GPU-needing call must fail named.
const { dr } = await import('dopplerx/compat');
try { await dr.load('qwen3-0.8b'); console.log('UNEXPECTED: load resolved'); process.exitCode = 1; }
catch (e) { console.log('no-gpu load failed:', e.code ?? e.name, '-', String(e.message).slice(0, 160)); }
