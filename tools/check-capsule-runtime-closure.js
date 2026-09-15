#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256Hex } from '../src/utils/sha256.js';
import { createJavaScriptDependencyGraph } from './lib/javascript-dependency-graph.js';

export async function buildRuntimeClosure(repoRoot, policy) {
  if (policy.schema !== 'doppler.runtime-closure-policy/v1') throw new Error('Invalid runtime closure policy.');
  const graph = createJavaScriptDependencyGraph();
  const files = new Map(), dynamicImports = [], observedRules = new Set();
  async function collect(file) {
    const normalized = path.resolve(file);
    if (files.has(normalized)) return;
    const relative = path.relative(repoRoot, normalized).replaceAll('\\', '/');
    if (relative.startsWith('../')) throw new Error(`Runtime dependency escapes repository: ${relative}`);
    const node = await graph.read(normalized);
    const bytes = Buffer.from(node.source);
    const record = { path: relative, sizeBytes: bytes.length, hash: `sha256:${sha256Hex(bytes)}` };
    files.set(normalized, record);
    for (const edge of node.unresolved) {
      const rule = policy.dynamicImports.find(item => item.path === relative && item.expression === edge.expression);
      if (!rule || rule.sourceHash !== record.hash || !rule.reason || rule.environment !== 'node'
        || !rule.specifier?.startsWith('node:')) {
        throw new Error(`Unresolved dynamic import requires a reviewed closure rule: ${relative}:${edge.line} import(${edge.expression})`);
      }
      observedRules.add(rule);
      dynamicImports.push({ path: relative, ...edge, specifier: rule.specifier, environment: rule.environment });
    }
    for (const { specifier, kind } of node.edges) {
      if (kind === 'type' || !specifier.startsWith('.')) continue;
      await collect(path.resolve(path.dirname(normalized), specifier.split(/[?#]/, 1)[0]));
    }
  }
  await collect(path.join(repoRoot, policy.entrypoint));
  if (observedRules.size !== policy.dynamicImports.length) throw new Error('Runtime closure policy contains stale dynamic import rules.');
  const records = [...files.values()].sort((a, b) => a.path.localeCompare(b.path));
  const forbiddenFiles = records.filter(record => policy.forbiddenPatterns.some(segment => `/${record.path}`.includes(segment)))
    .map(record => record.path);
  return {
    schema: 'doppler.capsule-runtime-closure/v1', entrypoint: policy.entrypoint,
    fileCount: records.length, sourceBytes: records.reduce((sum, record) => sum + record.sizeBytes, 0),
    forbiddenPatterns: policy.forbiddenPatterns,
    dynamicImports: dynamicImports.sort((a, b) => a.path.localeCompare(b.path) || a.line - b.line),
    forbiddenFiles, files: records, passed: forbiddenFiles.length === 0,
  };
}

async function main() {
  const repoRoot = path.resolve(import.meta.dirname, '..');
  const receiptPath = path.join(repoRoot, 'reports/capsule-runtime/runtime-closure.json');
  const policy = JSON.parse(await fs.readFile(path.join(repoRoot, 'tools/policies/runtime-closure-policy.json'), 'utf8'));
  const receipt = await buildRuntimeClosure(repoRoot, policy);
  if (process.argv.includes('--write')) {
    await fs.mkdir(path.dirname(receiptPath), { recursive: true });
    await fs.writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
  } else {
    const checkedIn = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
    if (JSON.stringify(checkedIn) !== JSON.stringify(receipt)) {
      throw new Error('Capsule runtime closure receipt is stale. Run npm run runtime:closure:sync.');
    }
  }
  if (!receipt.passed) throw new Error(`Capsule runtime closure contains forbidden files: ${receipt.forbiddenFiles.join(', ')}`);
  console.log(`Capsule runtime closure passed: ${receipt.fileCount} files, ${receipt.sourceBytes} source bytes.`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
