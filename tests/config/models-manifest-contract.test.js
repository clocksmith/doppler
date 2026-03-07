import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

function collectManifestPaths(rootDir) {
  const out = [];
  function walk(currentDir) {
    for (const entry of readdirSync(currentDir, { withFileTypes: true })) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
        continue;
      }
      if (entry.isFile() && entry.name === 'manifest.json') {
        out.push(fullPath);
      }
    }
  }
  walk(rootDir);
  out.sort((left, right) => left.localeCompare(right));
  return out;
}

function loadJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

for (const manifestPath of collectManifestPaths(path.join(process.cwd(), 'models'))) {
  const manifest = loadJson(manifestPath);
  const label = path.relative(process.cwd(), manifestPath);
  const inference = manifest?.inference;
  assert.ok(inference && typeof inference === 'object', `${label}: inference is required`);

  if (inference.schema !== 'doppler.execution/v0') {
    continue;
  }

  const sessionDefaults = inference.sessionDefaults;
  assert.ok(sessionDefaults && typeof sessionDefaults === 'object', `${label}: execution-v0 manifests require sessionDefaults`);
  assert.ok(
    sessionDefaults.compute?.defaults && typeof sessionDefaults.compute.defaults === 'object',
    `${label}: execution-v0 manifests require sessionDefaults.compute.defaults`
  );
  assert.notEqual(
    sessionDefaults.kvcache,
    undefined,
    `${label}: execution-v0 manifests require explicit sessionDefaults.kvcache`
  );
  assert.notEqual(
    sessionDefaults.decodeLoop,
    undefined,
    `${label}: execution-v0 manifests require explicit sessionDefaults.decodeLoop`
  );
  assert.ok(
    Array.isArray(inference.execution?.steps),
    `${label}: execution-v0 manifests require execution.steps`
  );
}

console.log('models-manifest-contract.test: ok');
