#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { buildDependencyGraph, dependencyPolicyErrors } from './lib/module-dependencies.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const DEFAULT_ENTRIES = Object.freeze([
  'src/index-browser.js',
  'src/tooling-exports.browser.js',
]);
const LOCAL_EXTENSIONS = Object.freeze(['.js', '.cjs']);

// Modules intentionally loaded only on Node via dynamic import (guarded by
// isNodeRuntime() / process.versions?.node). Their own node:* imports are
// therefore expected and safe from a browser bundle's static perspective.
const ALLOWED_NODE_GATED_MODULES = new Set([
  'src/client/runtime/node-quickstart-cache.js',
  'src/storage/artifact-storage-context.js',
  'src/tooling/browser-harness/model-initialization.js',
  'src/tooling/node-source-runtime.js',
  'src/tooling/browser-harness/model-resolution.js',
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

async function scanImportGraph(entryFile) {
  const declarations = JSON.parse(await fs.readFile(path.join(ROOT_DIR, 'tools/policies/module-dependencies.json'), 'utf8'));
  const entryAbsolute = path.resolve(ROOT_DIR, entryFile);
  const graphRoot = path.relative(ROOT_DIR, entryAbsolute).startsWith('..') ? path.dirname(entryAbsolute) : ROOT_DIR;
  const { graph, diagnostics } = await buildDependencyGraph(graphRoot, [entryAbsolute], declarations);
  const pending = [entryAbsolute];
  const seen = new Set();
  const issues = [];

  while (pending.length > 0) {
    const currentFile = pending.pop();
    if (!currentFile || seen.has(currentFile)) continue;
    seen.add(currentFile);

    let node;
    try {
      node = { edges: (graph.get(currentFile) ?? []).map(edge => ({ ...edge,
        kind: edge.kind === 'forward' ? 'static' : edge.kind })) };
      for (const diagnostic of dependencyPolicyErrors(diagnostics.filter(item => item.file === path.relative(graphRoot, currentFile).split(path.sep).join('/')))) {
        issues.push(`${diagnostic.kind}: ${diagnostic.file}:${diagnostic.line}: ${diagnostic.expression}`);
      }
    } catch (error) {
      issues.push(`unreadable module: ${toRepoPath(currentFile)} (${error?.message || 'unknown error'})`);
      continue;
    }

    const staticSpecs = node.edges.filter(edge => edge.kind === 'static').map(edge => edge.specifier);
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

    const dynamicSpecs = node.edges.filter(edge => edge.kind === 'dynamic').map(edge => edge.specifier);
    for (const specifier of dynamicSpecs) {
      if (!isLocalSpecifier(specifier)) continue;
      const resolved = await resolveLocalPath(currentFile, specifier);
      if (!resolved) {
        issues.push(`unresolved dynamic import: ${toRepoPath(currentFile)} -> ${specifier}`);
        continue;
      }
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
