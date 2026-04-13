import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { resolveLocalSourceRuntimePathFromModelUrl } = await import('../../src/inference/browser-harness-model-helpers.js');

const fixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-source-model-url-'));

try {
  const taskPath = path.join(fixtureDir, 'gemma-4-E2B-it-web.task');
  writeFileSync(taskPath, 'fixture');
  assert.equal(
    await resolveLocalSourceRuntimePathFromModelUrl(pathToFileURL(taskPath).href),
    taskPath
  );

  const rdrrDir = path.join(fixtureDir, 'rdrr');
  mkdirSync(rdrrDir);
  writeFileSync(path.join(rdrrDir, 'manifest.json'), '{}');
  assert.equal(
    await resolveLocalSourceRuntimePathFromModelUrl(pathToFileURL(rdrrDir).href),
    null
  );

  const safetensorsDir = path.join(fixtureDir, 'safetensors');
  mkdirSync(safetensorsDir);
  writeFileSync(path.join(safetensorsDir, 'config.json'), '{}');
  writeFileSync(path.join(safetensorsDir, 'model.safetensors'), 'fixture');
  assert.equal(
    await resolveLocalSourceRuntimePathFromModelUrl(pathToFileURL(safetensorsDir).href),
    safetensorsDir
  );

  const taskDir = path.join(fixtureDir, 'task-dir');
  mkdirSync(taskDir);
  writeFileSync(path.join(taskDir, 'gemma-4-E2B-it-web.task'), 'fixture');
  assert.equal(
    await resolveLocalSourceRuntimePathFromModelUrl(pathToFileURL(taskDir).href),
    taskDir
  );
} finally {
  rmSync(fixtureDir, { recursive: true, force: true });
}

console.log('browser-harness-local-source-model-url.test: ok');
