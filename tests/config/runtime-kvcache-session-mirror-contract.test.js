import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { createDopplerConfig } from '../../src/config/schema/index.js';

const LIVE_KVCACHE_KEYS = [
  'layout',
  'maxSeqLen',
  'pageSize',
  'kvDtype',
  'tiering',
  'quantization',
  'bdpaVocabSize',
  'gpuPagedFallbackMaxSeqLen',
  'forceF32Softcap',
  'windowSize',
];

async function walkJsonFiles(rootDir) {
  const entries = await fs.readdir(rootDir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walkJsonFiles(fullPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(fullPath);
    }
  }
  return files;
}

const runtimeConfigRoot = path.resolve('src/config/runtime');
const runtimeFiles = await walkJsonFiles(runtimeConfigRoot);
const defaultRuntime = createDopplerConfig().runtime;
const defaultTopLevelKV = defaultRuntime.inference.kvcache;
const defaultSessionKV = defaultRuntime.inference.session.kvcache;

for (const key of LIVE_KVCACHE_KEYS) {
  if (!Object.prototype.hasOwnProperty.call(defaultTopLevelKV, key)) {
    continue;
  }
  assert.deepEqual(
    defaultSessionKV[key] ?? null,
    defaultTopLevelKV[key] ?? null,
    `schema default runtime must mirror inference.kvcache.${key} into inference.session.kvcache.${key} for the live KV path.`
  );
}

for (const filePath of runtimeFiles) {
  const raw = await fs.readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw);
  const runtime = parsed.runtime ?? parsed;
  const topLevelKV = runtime?.inference?.kvcache ?? null;
  const sessionKV = runtime?.inference?.session?.kvcache ?? null;
  if (!topLevelKV || !sessionKV) {
    continue;
  }

  for (const key of LIVE_KVCACHE_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(topLevelKV, key)) {
      continue;
    }
    assert.deepEqual(
      sessionKV[key] ?? null,
      topLevelKV[key] ?? null,
      `${path.relative(process.cwd(), filePath)} must mirror inference.kvcache.${key} into inference.session.kvcache.${key} for the live KV path.`
    );
  }
}

console.log('runtime-kvcache-session-mirror-contract.test: ok');
