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
const LOCAL_EXTENSIONS = Object.freeze(['.js', '.cjs']);
const IMPORT_EXPORT_REGEX = /\b(?:import|export)\s+(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/g;
const DYNAMIC_IMPORT_REGEX = /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g;

// Modules intentionally loaded only on Node via dynamic import (guarded by
// isNodeRuntime() / process.versions?.node). Their own node:* imports are
// therefore expected and safe from a browser bundle's static perspective.
const ALLOWED_NODE_GATED_MODULES = new Set([
  'src/client/runtime/node-quickstart-cache.js',
  'src/storage/artifact-storage-context.js',
  'src/inference/browser-harness-model-helpers.js',
  'src/client/runtime/lora.js',
  'src/tooling/node-webgpu.js',
  'src/experimental/adapters/litert-runtime-bundle-node.js',
  'src/experimental/adapters/source-runtime-bundle-node.js',
  'src/experimental/adapters/source-runtime-bundle-node-dispatch.js',
]);

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

// Static imports are always followed. Dynamic imports are followed too, but
// their targets can opt out of graph scanning by appearing in
// ALLOWED_NODE_GATED_MODULES — those are the runtime-guarded Node bridges
// (isNodeRuntime() / process.versions?.node) whose node:* imports are
// intentionally unreachable from browser code at runtime.
function collectStaticSpecifiers(source) {
  const specifiers = [];
  IMPORT_EXPORT_REGEX.lastIndex = 0;
  for (;;) {
    const match = IMPORT_EXPORT_REGEX.exec(source);
    if (!match) break;
    if (typeof match[1] === 'string' && match[1].length > 0) {
      specifiers.push(match[1]);
    }
  }
  return specifiers;
}

function collectDynamicSpecifiers(source) {
  const specifiers = [];
  DYNAMIC_IMPORT_REGEX.lastIndex = 0;
  for (;;) {
    const match = DYNAMIC_IMPORT_REGEX.exec(source);
    if (!match) break;
    if (typeof match[1] === 'string' && match[1].length > 0) {
      specifiers.push(match[1]);
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

    const staticSpecs = collectStaticSpecifiers(source);
    for (const specifier of staticSpecs) {
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
      if (!seen.has(resolved)) pending.push(resolved);
    }

    const dynamicSpecs = collectDynamicSpecifiers(source);
    for (const specifier of dynamicSpecs) {
      if (!isLocalSpecifier(specifier)) continue;
      const resolved = await resolveLocalPath(currentFile, specifier);
      if (!resolved) continue;
      const repoPath = toRepoPath(resolved);
      if (ALLOWED_NODE_GATED_MODULES.has(repoPath)) continue;
      if (!seen.has(resolved)) pending.push(resolved);
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
