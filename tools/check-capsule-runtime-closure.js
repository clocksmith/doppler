#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256Hex } from '../src/utils/sha256.js';
import { buildDependencyGraph, collectReachable } from './lib/module-dependencies.js';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const entrypoint = path.join(repoRoot, 'src/capsule-runtime.js');
const receiptPath = path.join(repoRoot, 'reports/capsule-runtime/runtime-closure.json');
const declarations = JSON.parse(await fs.readFile(path.join(repoRoot, 'tools/policies/module-dependencies.json'), 'utf8'));
const forbidden = [
  '/src/converter/', '/src/training/', '/src/experimental/', '/src/models/',
  '/src/tooling/', '/src/cli/', '/src/client/provider.js', '/src/config/conversion/',
];

const { graph, diagnostics } = await buildDependencyGraph(repoRoot, [entrypoint], declarations);
if (diagnostics.length) throw new Error(`Runtime closure requires explicit dependencies: ${JSON.stringify(diagnostics)}`);
const files = collectReachable(graph, [entrypoint]);
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
  schema: 'doppler.capsule-runtime-closure/v1',
  entrypoint: 'src/capsule-runtime.js',
  scope: 'injected-runtime-core; host-supplied executors are inventoried separately',
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
    throw new Error('Capsule runtime closure receipt is stale. Run npm run runtime:closure:sync.');
  }
}
if (!receipt.passed) throw new Error(`Capsule runtime closure contains forbidden files: ${forbiddenFiles.join(', ')}`);
console.log(`Capsule runtime closure passed: ${receipt.fileCount} files, ${receipt.sourceBytes} source bytes.`);
