#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const FILE_RULES = [
  {
    file: 'src/index.js',
    allowed: new Set(['./version.js', './client/doppler-api.js', './client/provider.js']),
    forbidden: [
      'export * from',
      './tooling-exports',
      './loader/',
      './loaders/',
      './generation/',
      './experimental/orchestration/',
      './inference/',
      './gpu/',
      './experimental/adapters/',
      './storage/',
      './config/',
      './formats/',
    ],
  },
  {
    file: 'src/index.d.ts',
    allowed: new Set(['./version.js', './client/doppler-api.js', './client/provider.js']),
    forbidden: [
      'export * from',
      './tooling-exports',
      './loader/',
      './loaders/',
      './generation/',
      './experimental/orchestration/',
      './inference/',
      './gpu/',
      './experimental/adapters/',
      './storage/',
      './config/',
      './formats/',
    ],
  },
  {
    file: 'src/index-browser.js',
    allowed: new Set(['./version.js', './client/doppler-api.browser.js']),
    forbidden: [
      'export * from',
      './tooling-exports',
      './loader/',
      './loaders/',
      './generation/',
      './experimental/orchestration/',
      './inference/',
      './gpu/',
      './experimental/adapters/',
      './storage/',
      './config/',
      './formats/',
    ],
  },
  {
    file: 'src/index-browser.d.ts',
    allowed: new Set(['./version.js', './client/doppler-api.browser.js']),
    forbidden: [
      'export * from',
      './tooling-exports',
      './loader/',
      './loaders/',
      './generation/',
      './experimental/orchestration/',
      './inference/',
      './gpu/',
      './experimental/adapters/',
      './storage/',
      './config/',
      './formats/',
    ],
  },
  {
    file: 'src/generation/index.js',
    allowed: new Set(['../inference/pipelines/text.js']),
    forbidden: [
      'export * from',
      '../inference/pipelines/text/config.js',
      '../inference/pipelines/text/init.js',
      '../inference/pipelines/text/model-load.js',
      '../inference/pipelines/structured/',
      '../inference/pipelines/energy-head/',
      '../gpu/',
      '../experimental/adapters/',
    ],
  },
  {
    file: 'src/generation/index.d.ts',
    allowed: new Set([
      '../inference/pipelines/text.js',
      '../inference/pipelines/text/config.js',
      '../inference/pipelines/text/sampling.js',
      '../inference/pipelines/text/lora-types.js',
    ]),
    forbidden: [
      'export * from',
      '../inference/pipelines/text/init.js',
      '../inference/pipelines/text/model-load.js',
      '../inference/pipelines/structured/',
      '../inference/pipelines/energy-head/',
      '../gpu/',
      '../experimental/adapters/',
    ],
  },
];

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function readText(relativePath) {
  return fs.readFile(path.join(ROOT_DIR, relativePath), 'utf8');
}

function collectLocalSpecifiers(source) {
  const specifiers = new Set();
  const patterns = [
    /\bimport\s*(?:type\s*)?(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/g,
    /\bexport\s*(?:type\s*)?(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/g,
    /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
  ];
  for (const pattern of patterns) {
    pattern.lastIndex = 0;
    for (;;) {
      const match = pattern.exec(source);
      if (!match) break;
      specifiers.add(match[1]);
    }
  }
  return specifiers;
}

function isRelativeSpecifier(specifier) {
  return specifier.startsWith('./') || specifier.startsWith('../');
}

function validateFile(relativePath, rule, source) {
  const specifiers = collectLocalSpecifiers(source);
  for (const specifier of specifiers) {
    if (!isRelativeSpecifier(specifier)) {
      continue;
    }
    const isAllowed = rule.allowed.has(specifier);
    const isForbidden = rule.forbidden.some((token) => specifier.includes(token));
    assert(
      isAllowed,
      `${relativePath} exposes undeclared local specifier "${specifier}". Allowed: ${Array.from(rule.allowed).join(', ')}.`
    );
    assert(
      isAllowed || !isForbidden,
      `${relativePath} exposes forbidden local specifier "${specifier}".`
    );
  }

  for (const token of rule.forbidden) {
    assert(
      !source.includes(token),
      `${relativePath} contains forbidden boundary token "${token}".`
    );
  }
}

function normalizePackageTarget(target) {
  return target.startsWith('./') ? target.slice(2) : target;
}

function assertPackageTargetShape(target, label, options = {}) {
  assert(typeof target === 'string' && target.trim(), `${label} must be a non-empty string target.`);
  if (options.requireNodeExportPrefix) {
    assert(target.startsWith('./'), `${label} must use a package export target beginning with "./".`);
  }
  const normalized = normalizePackageTarget(target);
  assert(!path.isAbsolute(normalized), `${label} must be repo-relative.`);
  assert(!normalized.includes('\\'), `${label} must use forward slashes.`);
  assert(!normalized.split('/').includes('..'), `${label} must not traverse upward.`);
  return normalized;
}

async function assertPackageTargetExists(target, label, options = {}) {
  const normalized = assertPackageTargetShape(target, label, options);
  try {
    await fs.stat(path.join(ROOT_DIR, normalized));
  } catch {
    throw new Error(`${label} target does not exist: ${target}`);
  }
}

async function validatePackageExportTarget(target, label) {
  if (typeof target === 'string') {
    await assertPackageTargetExists(target, label, { requireNodeExportPrefix: true });
    return;
  }
  if (Array.isArray(target)) {
    for (let i = 0; i < target.length; i += 1) {
      await validatePackageExportTarget(target[i], `${label}[${i}]`);
    }
    return;
  }
  assert(target && typeof target === 'object', `${label} must be a string, array, or condition object.`);
  for (const [key, value] of Object.entries(target)) {
    await validatePackageExportTarget(value, `${label}.${key}`);
  }
}

async function validatePackageExportTargets(exportsField) {
  for (const [exportKey, target] of Object.entries(exportsField)) {
    await validatePackageExportTarget(target, `package.json exports ${exportKey}`);
  }
}

async function validatePackageBinTargets(packageJson) {
  const bins = packageJson.bin || {};
  if (typeof bins === 'string') {
    await assertPackageTargetExists(bins, 'package.json bin');
    return;
  }
  assert(bins && typeof bins === 'object' && !Array.isArray(bins), 'package.json bin must be a string or object.');
  for (const [name, target] of Object.entries(bins)) {
    await assertPackageTargetExists(target, `package.json bin ${name}`);
  }
}

async function validatePackageExports() {
  const packageJson = JSON.parse(await readText('package.json'));
  const exportsField = packageJson.exports || {};

  assert(exportsField['.'], 'package.json exports is missing the root entry.');
  assert(exportsField['./provider'], 'package.json exports must include ./provider.');
  assert(!exportsField['./internal'], 'package.json exports must not include ./internal.');
  assert(exportsField['./tooling'], 'package.json exports must include ./tooling.');
  assert(exportsField['./tooling-experimental'], 'package.json exports must include ./tooling-experimental.');
  assert(exportsField['./loaders'], 'package.json exports must include ./loaders.');
  assert(exportsField['./orchestration'], 'package.json exports must include ./orchestration.');
  assert(exportsField['./generation'], 'package.json exports must include ./generation.');

  const rootExport = exportsField['.'];
  assert(
    rootExport?.import === './src/index.js' && rootExport?.types === './src/index.d.ts',
    'package.json root export must point to src/index.js and src/index.d.ts.'
  );

  const providerExport = exportsField['./provider'];
  assert(
    providerExport?.import === './src/client/provider.js' && providerExport?.types === './src/client/provider.d.ts',
    'package.json ./provider export must point to src/client/provider.js and src/client/provider.d.ts.'
  );

  await validatePackageExportTargets(exportsField);
  await validatePackageBinTargets(packageJson);
}

async function main() {
  await validatePackageExports();

  for (const rule of FILE_RULES) {
    const source = await readText(rule.file);
    validateFile(rule.file, rule, source);
  }

  console.log('public boundary check passed');
}

try {
  await main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
