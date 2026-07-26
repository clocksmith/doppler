#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  computeCanonicalSha256,
  hashBytesSha256,
} from '../src/utils/canonical-hash.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = path.join(ROOT, 'demo', 'generated-shell-manifest.js');
const BUDGET_OUTPUT = path.join(ROOT, 'demo', 'generated-shell-budget.json');
const MAX_MODULES = 550;
const MAX_TOTAL_BYTES = 10_000_000;
const MODULE_ROOTS = [
  'demo/demo.js',
  'demo/sw.js',
];
const DECLARED_ASSETS = [
  'demo/index.html',
  'demo/pwa-manifest.json',
  'demo/favicon.svg',
  'demo/examples.json',
  'demo/styles/rd.css',
  'demo/styles/rd-tokens.css',
  'demo/styles/rd-primitives.css',
  'demo/styles/rd-components.css',
  'demo/ui/styles/app.css',
  'demo/ui/word-quality/styles.css',
  'demo/ui/xray/styles.css',
  'demo/assets/pwa/icon-192.png',
  'demo/assets/pwa/icon-512.png',
  'demo/assets/pwa/icon-maskable-512.png',
  'demo/assets/pwa/shortcut-new-96.png',
  'demo/assets/pwa/shortcut-xray-96.png',
  'demo/assets/pwa/screenshot-desktop.png',
  'src/config/runtime/profiles/default.json',
  'src/config/runtime/profiles/throughput.json',
  'src/config/runtime/profiles/verbose-trace.json',
  'src/config/runtime/profiles/production.json',
  'src/config/runtime/profiles/low-memory.json',
  'src/config/runtime/profiles/trace-layers.json',
];
const BARE_SPECIFIERS = new Map([
  ['doppler-gpu', 'src/index-browser.js'],
  ['doppler-gpu/tooling/runtime', 'src/tooling-exports/runtime.js'],
  ['doppler-gpu/tooling/evidence', 'src/tooling-exports/evidence.js'],
]);
const SPECIFIER_PATTERNS = [
  /\b(?:import|export)\s+(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/g,
  /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
];

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function resolveSpecifier(importer, specifier) {
  if (BARE_SPECIFIERS.has(specifier)) return BARE_SPECIFIERS.get(specifier);
  if (!specifier.startsWith('.')) return null;
  const base = path.resolve(ROOT, path.dirname(importer), specifier);
  const candidates = path.extname(base)
    ? [base]
    : [`${base}.js`, path.join(base, 'index.js')];
  for (const candidate of candidates) {
    if (await exists(candidate)) {
      return path.relative(ROOT, candidate).split(path.sep).join('/');
    }
  }
  throw new Error(`${importer}: unresolved local module ${specifier}`);
}

function collectSpecifiers(source) {
  const values = [];
  for (const pattern of SPECIFIER_PATTERNS) {
    pattern.lastIndex = 0;
    for (const match of source.matchAll(pattern)) values.push(match[1]);
  }
  return values;
}

async function collectModules() {
  const pending = [...MODULE_ROOTS];
  const seen = new Set();
  while (pending.length > 0) {
    const file = pending.pop();
    if (!file || seen.has(file)) continue;
    seen.add(file);
    const source = await fs.readFile(path.join(ROOT, file), 'utf8');
    for (const specifier of collectSpecifiers(source)) {
      const resolved = await resolveSpecifier(file, specifier);
      if (resolved && !seen.has(resolved)) pending.push(resolved);
    }
    for (const match of source.matchAll(/\bloadJson\(\s*['"]([^'"]+\.json)['"]/g)) {
      const candidate = path.resolve(ROOT, path.dirname(file), match[1]);
      if (await exists(candidate)) {
        seen.add(path.relative(ROOT, candidate).split(path.sep).join('/'));
      }
    }
  }
  return seen;
}

function renderManifest(files, digest) {
  const urls = files.map((file) => `/${file}`);
  return [
    `export const SHELL_MANIFEST_SCHEMA = 'doppler.demo-shell-manifest/v1';`,
    `export const SHELL_MANIFEST_DIGEST = '${digest}';`,
    `export const CACHE_NAME = 'doppler-demo-shell-${digest.slice(7, 23)}';`,
    `export const APP_SHELL = Object.freeze(${JSON.stringify(urls, null, 2)});`,
    '',
  ].join('\n');
}

async function main() {
  const checkOnly = process.argv.slice(2).includes('--check');
  const unsupported = process.argv.slice(2).filter((token) => token !== '--check');
  if (unsupported.length > 0) {
    throw new Error(`Unknown argument: ${unsupported[0]}`);
  }
  const modules = await collectModules();
  const files = [...new Set([...modules, ...DECLARED_ASSETS])].sort();
  for (const file of files) {
    if (!await exists(path.join(ROOT, file))) {
      throw new Error(`Declared demo shell asset does not exist: ${file}`);
    }
  }
  const contentEntries = await Promise.all(
    files
      .filter((file) => file !== 'demo/generated-shell-manifest.js')
      .map(async (file) => ({
        path: file,
        digest: hashBytesSha256(await fs.readFile(path.join(ROOT, file))),
      }))
  );
  const digest = computeCanonicalSha256({
    schema: 'doppler.demo-shell-manifest/v1',
    entries: contentEntries,
  });
  const moduleCount = files.filter((file) => file.endsWith('.js')).length;
  const manifestSource = renderManifest(files, digest);
  const totalBytes = (await Promise.all(
    files.map(async (file) => (
      file === 'demo/generated-shell-manifest.js'
        ? Buffer.byteLength(manifestSource)
        : (await fs.stat(path.join(ROOT, file))).size
    ))
  )).reduce((sum, size) => sum + size, 0);
  if (moduleCount > MAX_MODULES) {
    throw new Error(`Demo shell module budget exceeded: ${moduleCount} > ${MAX_MODULES}`);
  }
  if (totalBytes > MAX_TOTAL_BYTES) {
    throw new Error(`Demo shell byte budget exceeded: ${totalBytes} > ${MAX_TOTAL_BYTES}`);
  }
  const budgetSource = `${JSON.stringify({
    schema: 'doppler.demo-shell-budget-receipt/v1',
    shellManifestDigest: digest,
    fileCount: files.length,
    moduleCount,
    totalBytes,
    budgets: {
      maximumModules: MAX_MODULES,
      maximumTotalBytes: MAX_TOTAL_BYTES,
    },
    passed: true,
  }, null, 2)}\n`;
  if (checkOnly) {
    const [currentManifest, currentBudget] = await Promise.all([
      fs.readFile(OUTPUT, 'utf8'),
      fs.readFile(BUDGET_OUTPUT, 'utf8'),
    ]);
    if (currentManifest !== manifestSource || currentBudget !== budgetSource) {
      throw new Error('Generated demo shell evidence is stale; run npm run demo:shell:generate.');
    }
  } else {
    await fs.writeFile(OUTPUT, manifestSource, 'utf8');
    await fs.writeFile(BUDGET_OUTPUT, budgetSource, 'utf8');
  }
  console.log(`demo shell manifest: ${files.length} files, ${moduleCount} modules, ${totalBytes} bytes (${digest})`);
}

await main();
