#!/usr/bin/env node
// Keeps conversion configs, source-packages, and compiled manifests in sync with
// src/config/kernels/kernel-ref-digests.js. Run with --check to fail on drift.

import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
let checkOnly = false;
let sourceOnly = false;
let packageRootArgument = null;
const selectedFiles = [];
const usage = 'Usage: sync-conversion-kernel-digests.js [--check] [--source-only] '
  + '[--file <candidate-recipe.json>] [--package-root <installed-package-directory>]';
for (let index = 2; index < process.argv.length; index += 1) {
  const argument = process.argv[index];
  if (argument === '--check') checkOnly = true;
  else if (argument === '--source-only') sourceOnly = true;
  else if (['--file', '--package-root'].includes(argument)
    && process.argv[index + 1] && !process.argv[index + 1].startsWith('--')) {
    const value = path.resolve(process.argv[++index]);
    if (argument === '--file') selectedFiles.push(value);
    else if (packageRootArgument === null) packageRootArgument = value;
    else throw new Error(usage);
  } else throw new Error(usage);
}
if (packageRootArgument && (!checkOnly || selectedFiles.length > 0)) {
  throw new Error('--package-root requires an installed package directory and --check; it cannot select --file.');
}
const scanRoot = packageRootArgument ?? ROOT;
sourceOnly = sourceOnly || packageRootArgument !== null;
if (sourceOnly && selectedFiles.length > 0) {
  throw new Error('--source-only cannot select --file.');
}

const refPath = path.join(scanRoot, 'src/config/kernels/kernel-ref-digests.js');
const canonical = new Map();
if (packageRootArgument) {
  const registry = JSON.parse(fs.readFileSync(path.join(scanRoot, 'src/config/kernels/registry.json'), 'utf8'));
  for (const operation of Object.values(registry.operations)) {
    for (const variant of Object.values(operation.variants)) {
      const source = fs.readFileSync(path.join(scanRoot, 'src/gpu/kernels', variant.wgsl), 'utf8').replace(/\r\n/g, '\n');
      const entry = variant.entryPoint;
      canonical.set(`${variant.wgsl}#${entry}`, createHash('sha256').update(`${source}\n@@entry:${entry}`).digest('hex'));
    }
  }
} else {
  const refSource = fs.readFileSync(refPath, 'utf8');
  for (const match of refSource.matchAll(/"([^"]+#[^"]+)":\s*"([a-f0-9]+)"/g)) {
    canonical.set(match[1], match[2]);
  }
}

const ROOTS = [
  'src/config/conversion',
  'src/config/source-packages',
  'models/local',
];
if (canonical.size === 0) throw new Error(`No canonical kernel digests found in ${refPath}.`);
const scanDirectories = sourceOnly ? ROOTS.filter((value) => value !== 'models/local') : ROOTS;
if (sourceOnly) {
  for (const directory of scanDirectories) {
    if (!fs.statSync(path.join(scanRoot, directory)).isDirectory()) {
      throw new Error(`Missing package configuration directory: ${directory}`);
    }
  }
}

function walk(dir, acc = []) {
  const abs = path.join(scanRoot, dir);
  if (!fs.existsSync(abs)) return acc;
  for (const entry of fs.readdirSync(abs, { withFileTypes: true })) {
    const p = path.join(abs, entry.name);
    if (entry.isDirectory()) walk(path.relative(scanRoot, p), acc);
    else if (entry.name.endsWith('.json')) acc.push(p);
  }
  return acc;
}

const files = selectedFiles.length > 0 ? [...new Set(selectedFiles)] : scanDirectories.flatMap((directory) => walk(directory));

let drifted = 0;
const changedFiles = new Set();

for (const file of files) {
  let data;
  try { data = JSON.parse(fs.readFileSync(file, 'utf8')); } catch (error) {
    if (selectedFiles.length > 0 || sourceOnly) throw error;
    continue;
  }
  let changed = false;
  (function walkNode(node) {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { for (const child of node) walkNode(child); return; }
    if (typeof node.kernel === 'string' && typeof node.entry === 'string' && typeof node.digest === 'string') {
      const key = `${node.kernel}#${node.entry}`;
      const want = canonical.get(key);
      if (!want && selectedFiles.length > 0) throw new Error(`Unknown kernel reference "${key}" in ${file}.`);
      const have = node.digest.replace(/^sha256:/, '');
      if (want && want !== have) {
        drifted++;
        if (checkOnly) console.error(`${path.relative(scanRoot, file)}: ${key} has sha256:${have}; expected sha256:${want}`);
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
    const repair = sourceOnly
      ? 'Repair source with npm run kernels:conversion-digests:sync -- --source-only, then rebuild the package.'
      : 'Run: npm run kernels:conversion-digests:sync';
    console.error(`[kernels:conversion-digests:check] ${drifted} digest(s) drift from kernel-ref-digests.js. ${repair}`);
    process.exit(1);
  }
  console.log(`[kernels:conversion-digests:check] registered kernel digests match in ${files.length} configuration files`);
} else {
  console.log(`[kernels:conversion-digests:sync] updated ${drifted} digest(s) in ${changedFiles.size} file(s).`);
  for (const file of changedFiles) console.log(`  ${path.relative(scanRoot, file)}`);
}
