#!/usr/bin/env node
// Keeps conversion configs, source-packages, and compiled manifests in sync with
// src/config/kernels/kernel-ref-digests.js. Run with --check to fail on drift.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const refPath = path.join(ROOT, 'src/config/kernels/kernel-ref-digests.js');
const refSource = fs.readFileSync(refPath, 'utf8');
const canonical = new Map();
for (const match of refSource.matchAll(/"([^"]+#[^"]+)":\s*"([a-f0-9]+)"/g)) {
  canonical.set(match[1], match[2]);
}

const ROOTS = [
  'src/config/conversion',
  'src/config/source-packages',
  'models/local',
];

function walk(dir, acc = []) {
  const abs = path.join(ROOT, dir);
  if (!fs.existsSync(abs)) return acc;
  for (const entry of fs.readdirSync(abs, { withFileTypes: true })) {
    const p = path.join(abs, entry.name);
    if (entry.isDirectory()) walk(path.relative(ROOT, p), acc);
    else if (entry.name.endsWith('.json')) acc.push(p);
  }
  return acc;
}

const checkOnly = process.argv.includes('--check');
const files = ROOTS.flatMap((root) => walk(root));

let drifted = 0;
const changedFiles = new Set();

for (const file of files) {
  let data;
  try { data = JSON.parse(fs.readFileSync(file, 'utf8')); } catch { continue; }
  let changed = false;
  (function walkNode(node) {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { for (const child of node) walkNode(child); return; }
    if (typeof node.kernel === 'string' && typeof node.entry === 'string' && typeof node.digest === 'string') {
      const key = `${node.kernel}#${node.entry}`;
      const want = canonical.get(key);
      const have = node.digest.replace(/^sha256:/, '');
      if (want && want !== have) {
        drifted++;
        if (!checkOnly) {
          node.digest = `sha256:${want}`;
          changed = true;
        }
      }
    }
    for (const value of Object.values(node)) walkNode(value);
  })(data);
  if (changed) {
    changedFiles.add(file);
    fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
  }
}

if (checkOnly) {
  if (drifted > 0) {
    console.error(`[kernels:conversion-digests:check] ${drifted} digest(s) drift from kernel-ref-digests.js. Run: npm run kernels:conversion-digests:sync`);
    process.exit(1);
  }
  console.log('[kernels:conversion-digests:check] all conversion/source-package/manifest digests match kernel-ref-digests.js');
} else {
  console.log(`[kernels:conversion-digests:sync] updated ${drifted} digest(s) in ${changedFiles.size} file(s).`);
  for (const file of changedFiles) console.log(`  ${path.relative(ROOT, file)}`);
}
