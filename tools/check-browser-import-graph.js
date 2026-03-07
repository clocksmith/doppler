#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const DEFAULT_ENTRIES = Object.freeze([
  'src/index-browser.js',
  'src/tooling-exports.browser.js',
]);
const LOCAL_EXTENSIONS = Object.freeze(['.js', '.mjs', '.cjs']);
const IMPORT_EXPORT_REGEX = /\b(?:import|export)\s+(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/g;
const DYNAMIC_IMPORT_REGEX = /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g;

function toRepoPath(filePath) {
  return path.relative(ROOT_DIR, filePath).split(path.sep).join('/');
}

function stripQueryAndHash(specifier) {
  const queryIndex = specifier.indexOf('?');
  const hashIndex = specifier.indexOf('#');
  let end = specifier.length;
  if (queryIndex >= 0) end = Math.min(end, queryIndex);
  if (hashIndex >= 0) end = Math.min(end, hashIndex);
  return specifier.slice(0, end);
}

function isLocalSpecifier(specifier) {
  return specifier.startsWith('./')
    || specifier.startsWith('../')
    || specifier.startsWith('/');
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function resolveLocalPath(importerPath, rawSpecifier) {
  const specifier = stripQueryAndHash(rawSpecifier);
  const basePath = specifier.startsWith('/')
    ? path.resolve(ROOT_DIR, `.${specifier}`)
    : path.resolve(path.dirname(importerPath), specifier);

  const candidates = [basePath];
  const ext = path.extname(basePath);
  if (!ext) {
    for (const candidateExt of LOCAL_EXTENSIONS) {
      candidates.push(`${basePath}${candidateExt}`);
      candidates.push(path.join(basePath, `index${candidateExt}`));
    }
  }

  for (const candidate of candidates) {
    if (!await pathExists(candidate)) continue;
    const stats = await fs.stat(candidate);
    if (stats.isFile()) return candidate;
  }
  return null;
}

function collectSpecifiers(source) {
  const specifiers = [];
  for (const regex of [IMPORT_EXPORT_REGEX, DYNAMIC_IMPORT_REGEX]) {
    regex.lastIndex = 0;
    for (;;) {
      const match = regex.exec(source);
      if (!match) break;
      if (typeof match[1] === 'string' && match[1].length > 0) {
        specifiers.push(match[1]);
      }
    }
  }
  return specifiers;
}

async function scanImportGraph(entryFile) {
  const entryAbsolute = path.resolve(ROOT_DIR, entryFile);
  const pending = [entryAbsolute];
  const seen = new Set();
  const issues = [];

  while (pending.length > 0) {
    const currentFile = pending.pop();
    if (!currentFile || seen.has(currentFile)) continue;
    seen.add(currentFile);

    let source = '';
    try {
      source = await fs.readFile(currentFile, 'utf-8');
    } catch (error) {
      issues.push(`unreadable module: ${toRepoPath(currentFile)} (${error?.message || 'unknown error'})`);
      continue;
    }

    const specifiers = collectSpecifiers(source);
    for (const specifier of specifiers) {
      if (specifier.startsWith('node:')) {
        issues.push(`node:* specifier found: ${toRepoPath(currentFile)} -> ${specifier}`);
        continue;
      }
      if (!isLocalSpecifier(specifier)) continue;

      const resolved = await resolveLocalPath(currentFile, specifier);
      if (!resolved) {
        issues.push(`unresolved local import: ${toRepoPath(currentFile)} -> ${specifier}`);
        continue;
      }
      if (!seen.has(resolved)) {
        pending.push(resolved);
      }
    }
  }

  return issues;
}

async function main() {
  const entryFiles = process.argv.length > 2
    ? process.argv.slice(2)
    : DEFAULT_ENTRIES;
  let failed = false;

  for (const entryFile of entryFiles) {
    const issues = await scanImportGraph(entryFile);
    if (issues.length > 0) {
      console.error(`browser import graph check failed (${entryFile}):`);
      for (const issue of issues) {
        console.error(`- ${issue}`);
      }
      failed = true;
      continue;
    }
    console.log(`browser import graph check passed (${entryFile})`);
  }

  if (failed) {
    process.exitCode = 1;
  }
}

await main();
