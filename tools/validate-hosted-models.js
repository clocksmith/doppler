#!/usr/bin/env node

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(new URL(import.meta.url)));
const repoRoot = path.resolve(scriptDir, '..');
const modelsDir = path.resolve(repoRoot, 'models');
const curatedDir = path.resolve(modelsDir, 'curated');
const catalogPath = path.resolve(modelsDir, 'catalog.json');
const firebaseConfigPath = path.resolve(repoRoot, 'firebase.json');

function toPosix(relPath) {
  return relPath.split(path.sep).join('/');
}

function fail(message) {
  console.error(`[models-validate] ${message}`);
  process.exit(1);
}

async function readJson(filePath, label) {
  let text;
  try {
    text = await fs.readFile(filePath, 'utf8');
  } catch (error) {
    fail(`Missing required file: ${label} (${error.message})`);
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    fail(`Invalid JSON in ${label}: ${error.message}`);
  }
}

async function collectFiles(rootDir, out = []) {
  const entries = await fs.readdir(rootDir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      await collectFiles(fullPath, out);
      continue;
    }
    if (entry.isFile()) {
      out.push(fullPath);
    }
  }
  return out;
}

async function collectImmediateDirs(rootDir) {
  try {
    const entries = await fs.readdir(rootDir, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort((a, b) => a.localeCompare(b));
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return [];
    }
    throw error;
  }
}

function normalizeCatalogBase(baseUrl) {
  if (typeof baseUrl !== 'string') return null;
  const trimmed = baseUrl.trim();
  if (trimmed.length === 0) return null;
  if (/^[a-z]+:\/\//i.test(trimmed)) return null;
  let normalized = trimmed.replace(/\\/g, '/');
  while (normalized.startsWith('./')) {
    normalized = normalized.slice(2);
  }
  if (normalized.startsWith('/')) {
    normalized = normalized.slice(1);
  }
  if (normalized.startsWith('models/')) {
    normalized = normalized.slice('models/'.length);
  }
  if (normalized.includes('..')) return null;
  return normalized;
}

function decodePathToken(token) {
  try {
    return decodeURIComponent(token);
  } catch {
    return token;
  }
}

function resolveCatalogCuratedDir(entry) {
  if (!entry || typeof entry !== 'object') return null;
  const modelId = typeof entry.modelId === 'string' ? entry.modelId.trim() : '';
  if (!modelId) return null;
  const normalizedBase = normalizeCatalogBase(
    typeof entry.baseUrl === 'string' && entry.baseUrl.trim().length > 0
      ? entry.baseUrl
      : `./curated/${encodeURIComponent(modelId)}`
  );
  if (!normalizedBase || !normalizedBase.startsWith('curated/')) return null;
  const suffix = normalizedBase.slice('curated/'.length);
  const firstToken = suffix.split('/')[0];
  if (!firstToken) return null;
  return decodePathToken(firstToken);
}

function requireIgnorePattern(ignoreList, pattern) {
  if (!ignoreList.includes(pattern)) {
    fail(`firebase.json hosting.ignore must include "${pattern}".`);
  }
}

async function main() {
  const catalog = await readJson(catalogPath, 'models/catalog.json');
  const firebaseConfig = await readJson(firebaseConfigPath, 'firebase.json');

  const files = await collectFiles(modelsDir);
  const disallowed = [];
  for (const filePath of files.sort((a, b) => a.localeCompare(b))) {
    const relPath = toPosix(path.relative(modelsDir, filePath));
    if (
      relPath === 'catalog.json' ||
      relPath === 'README.md' ||
      relPath.startsWith('curated/') ||
      relPath.startsWith('local/') ||
      relPath.startsWith('converted/')
    ) {
      continue;
    }
    disallowed.push(relPath);
  }
  if (disallowed.length > 0) {
    fail(`Disallowed files found under models/: ${disallowed.join(', ')}`);
  }

  const hostingIgnore = firebaseConfig?.hosting?.ignore;
  if (!Array.isArray(hostingIgnore)) {
    fail('firebase.json is missing hosting.ignore array.');
  }
  requireIgnorePattern(hostingIgnore, 'models/local/**');
  requireIgnorePattern(hostingIgnore, 'models/converted/**');
  if (hostingIgnore.includes('models/curated/**')) {
    fail('firebase.json must not ignore all curated models (models/curated/**).');
  }

  const catalogModels = Array.isArray(catalog?.models) ? catalog.models : [];
  const catalogCuratedDirs = new Set();
  for (let i = 0; i < catalogModels.length; i += 1) {
    const entry = catalogModels[i];
    const curatedSubdir = resolveCatalogCuratedDir(entry);
    if (!curatedSubdir) {
      fail(`catalog entry index ${i} must resolve to a curated baseUrl; got modelId=${entry?.modelId ?? '<missing>'}`);
    }
    catalogCuratedDirs.add(curatedSubdir);
  }

  const curatedSubdirs = await collectImmediateDirs(curatedDir);
  for (const dirName of curatedSubdirs) {
    const manifestPath = path.join(curatedDir, dirName, 'manifest.json');
    try {
      await fs.access(manifestPath);
    } catch {
      fail(`Curated model "${dirName}" is missing manifest.json`);
    }
  }

  for (const dirName of catalogCuratedDirs) {
    if (!curatedSubdirs.includes(dirName)) {
      fail(`catalog references missing curated model directory: models/curated/${dirName}`);
    }
    if (hostingIgnore.includes(`models/curated/${dirName}/**`)) {
      fail(`catalog model is currently excluded by firebase ignore: models/curated/${dirName}/**`);
    }
  }

  for (const dirName of curatedSubdirs) {
    if (catalogCuratedDirs.has(dirName)) continue;
    const ignorePattern = `models/curated/${dirName}/**`;
    if (!hostingIgnore.includes(ignorePattern)) {
      fail(`non-catalog curated model must be excluded from hosting.ignore: ${ignorePattern}`);
    }
  }

  console.log('[models-validate] OK: hosted models are curated + cataloged only; non-curated trees are excluded.');
}

main().catch((error) => {
  console.error(error.message || String(error));
  process.exit(1);
});
