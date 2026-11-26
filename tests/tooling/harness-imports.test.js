// Every module tests/harness.html can load must exist. The harness imports lazily per mode, so a broken
// path only shows when that mode runs in a browser; this walks the literal import graph from the page instead.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const harness = path.join(root, 'tests', 'harness.html');

// Modules the serving host provides; Doppler does not ship them. Simulation mode needs the host simulator
// (the same contract src/inference/pipelines/text/initialization/init-emulation.js declares).
const HOST_PROVIDED = new Set(['/proto/simulator/index.js']);

const SPECIFIER = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])([^'"\n]+)\1/g;

function specifiers(source) {
  return [...source.matchAll(SPECIFIER)].map((match) => match[2]);
}

function resolve(specifier, fromFile) {
  if (specifier.startsWith('/')) return path.join(root, specifier);
  if (specifier.startsWith('.')) return path.resolve(path.dirname(fromFile), specifier);
  return null; // bare specifiers (node:, packages) are not the harness's own files
}

const missing = [];
const hostImports = new Set();
const seen = new Set();
const queue = [harness];
while (queue.length) {
  const file = queue.pop();
  if (seen.has(file)) continue;
  seen.add(file);
  for (const specifier of specifiers(fs.readFileSync(file, 'utf8'))) {
    if (HOST_PROVIDED.has(specifier)) {
      hostImports.add(specifier);
      continue;
    }
    const target = resolve(specifier, file);
    if (!target) continue;
    if (!fs.existsSync(target)) {
      missing.push(`${path.relative(root, file)} -> ${specifier}`);
      continue;
    }
    if (/\.(js|mjs)$/.test(target)) queue.push(target);
  }
}

assert.ok(seen.size > 50, `expected the harness to reach many modules, reached ${seen.size}`);
assert.deepEqual(missing, [], 'imports reachable from tests/harness.html that resolve to no file');
for (const specifier of HOST_PROVIDED) {
  assert.ok(hostImports.has(specifier), `${specifier} is listed as host-provided but nothing imports it; remove it`);
}
console.log(`harness-imports.test: ok (${seen.size} files)`);
