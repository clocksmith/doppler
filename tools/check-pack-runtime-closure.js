#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256Hex } from '../src/utils/sha256.js';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const entrypoint = path.join(repoRoot, 'src/pack-runtime.js');
const receiptPath = path.join(repoRoot, 'reports/pack-runtime/runtime-closure.json');
const forbidden = [
  '/src/converter/', '/src/training/', '/src/experimental/', '/src/models/',
  '/src/tooling/', '/src/cli/', '/src/client/provider.js', '/src/config/conversion/',
];
const importPattern = /(?:import|export)\s+(?:[^'";]*?\s+from\s+)?['"]([^'"]+)['"]/g;

async function collect(file, files) {
  const normalized = path.resolve(file);
  if (files.has(normalized)) return;
  files.add(normalized);
  const source = await fs.readFile(normalized, 'utf8');
  for (const match of source.matchAll(importPattern)) {
    const specifier = match[1];
    if (!specifier.startsWith('.')) continue;
    const dependency = path.resolve(path.dirname(normalized), specifier);
    await collect(dependency, files);
  }
}

const files = new Set();
await collect(entrypoint, files);
const records = [];
for (const file of [...files].sort()) {
  const source = await fs.readFile(file);
  records.push({
    path: path.relative(repoRoot, file).replaceAll('\\', '/'),
    sizeBytes: source.byteLength,
    hash: `sha256:${sha256Hex(source)}`,
  });
}
const forbiddenFiles = records.filter((record) => forbidden.some((segment) => (
  `/${record.path}`.includes(segment)
))).map((record) => record.path);
const receipt = {
  schema: 'doppler.pack-runtime-closure/v1',
  entrypoint: 'src/pack-runtime.js',
  fileCount: records.length,
  sourceBytes: records.reduce((total, record) => total + record.sizeBytes, 0),
  forbiddenPatterns: forbidden,
  forbiddenFiles,
  files: records,
  passed: forbiddenFiles.length === 0,
};
if (process.argv.includes('--write')) {
  await fs.mkdir(path.dirname(receiptPath), { recursive: true });
  await fs.writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
} else {
  const checkedIn = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
  if (JSON.stringify(checkedIn) !== JSON.stringify(receipt)) {
    throw new Error('Pack runtime closure receipt is stale. Run npm run runtime:closure:sync.');
  }
}
if (!receipt.passed) throw new Error(`Pack runtime closure contains forbidden files: ${forbiddenFiles.join(', ')}`);
console.log(`Pack runtime closure passed: ${receipt.fileCount} files, ${receipt.sourceBytes} source bytes.`);
