#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SOURCE_DIR = path.join(ROOT_DIR, 'src');
const SOURCE_IGNORE_PATH = path.join(SOURCE_DIR, '.npmignore');
const EXTRA_RUNTIME_ENTRIES = Object.freeze(['tools/convert-safetensors-node.js']);
const PACKAGE_RUNTIME_PREFIXES = Object.freeze([
  'src/',
  'tests/kernels/browser/',
  'tests/kernels/harness/',
  'tests/kernels/reference/',
]);
const PACKAGE_RUNTIME_FILES = Object.freeze(new Set(EXTRA_RUNTIME_ENTRIES));
const PUBLIC_RESOURCE_DIRS = Object.freeze([
  'src/config/conversion/',
  'src/config/platforms/',
  'src/config/runtime/',
]);
const REQUIRED_RESOURCE_FILES = Object.freeze([
  'models/catalog.json',
  'src/tooling/command-runner.html',
]);
const ALWAYS_IGNORED_SOURCE_PREFIXES = Object.freeze([
  'debug/reference/',
  'gpu/kernels/codegen/',
]);
const MODULE_SPECIFIER_PATTERNS = Object.freeze([
  /\b(?:import|export)\s+(?:type\s+)?(?:[^'";]*?\sfrom\s*)?['"]([^'"]+)['"]/gu,
  /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/gu,
]);
const RELATIVE_FILE_LITERAL_PATTERN = /['"`]((?:\.\.\/|\.\/)[^'"`\n]+\.(?:d\.ts|js|json|html|wgsl))['"`]/gu;

function normalizePath(filePath) {
  return path.relative(ROOT_DIR, filePath).split(path.sep).join('/');
}

function isLocalSpecifier(specifier) {
  return specifier.startsWith('./') || specifier.startsWith('../');
}

function stripQueryAndHash(specifier) {
  return specifier.split(/[?#]/u, 1)[0];
}

async function pathKind(filePath) {
  try {
    const stats = await fs.stat(filePath);
    if (stats.isFile()) return 'file';
    if (stats.isDirectory()) return 'directory';
  } catch {
    return null;
  }
  return null;
}

async function walkFiles(rootDir, predicate = () => true) {
  const output = [];
  async function walk(currentDir) {
    const entries = await fs.readdir(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        await walk(entryPath);
      } else if (entry.isFile() && predicate(entryPath)) {
        output.push(entryPath);
      }
    }
  }
  await walk(rootDir);
  return output;
}

function collectTargets(target, output) {
  if (typeof target === 'string') {
    output.add(target.startsWith('./') ? target.slice(2) : target);
    return;
  }
  if (Array.isArray(target)) {
    for (const entry of target) collectTargets(entry, output);
    return;
  }
  if (!target || typeof target !== 'object') return;
  for (const value of Object.values(target)) collectTargets(value, output);
}

function collectEntrypoints(packageJson) {
  const targets = new Set();
  collectTargets(packageJson.exports, targets);
  collectTargets(packageJson.bin, targets);
  if (typeof packageJson.main === 'string') targets.add(packageJson.main);
  if (typeof packageJson.types === 'string') targets.add(packageJson.types);
  const runtime = new Set(EXTRA_RUNTIME_ENTRIES);
  const types = new Set();
  for (const target of targets) {
    if (target.endsWith('.d.ts')) {
      types.add(target);
    } else if (target.endsWith('.js')) {
      runtime.add(target);
    }
  }
  return { runtime, types };
}

function collectModuleSpecifiers(source) {
  const output = new Set();
  for (const pattern of MODULE_SPECIFIER_PATTERNS) {
    pattern.lastIndex = 0;
    for (;;) {
      const match = pattern.exec(source);
      if (!match) break;
      output.add(match[1]);
    }
  }
  return output;
}

function collectRelativeFileLiterals(source) {
  const output = new Set();
  RELATIVE_FILE_LITERAL_PATTERN.lastIndex = 0;
  for (;;) {
    const match = RELATIVE_FILE_LITERAL_PATTERN.exec(source);
    if (!match) break;
    output.add(match[1]);
  }
  return output;
}

async function resolveRuntimeDependency(importerPath, rawSpecifier) {
  const specifier = stripQueryAndHash(rawSpecifier);
  if (!isLocalSpecifier(specifier)) return null;
  const basePath = path.resolve(path.dirname(importerPath), specifier);
  const candidates = path.extname(basePath)
    ? [basePath]
    : [basePath, `${basePath}.js`, path.join(basePath, 'index.js')];
  for (const candidate of candidates) {
    if (await pathKind(candidate) === 'file') return candidate;
  }
  return null;
}

async function resolveTypeDependency(importerPath, rawSpecifier) {
  const specifier = stripQueryAndHash(rawSpecifier);
  if (!isLocalSpecifier(specifier)) return null;
  const basePath = path.resolve(path.dirname(importerPath), specifier);
  const candidates = [];
  if (basePath.endsWith('.js')) {
    candidates.push(basePath.replace(/\.js$/u, '.d.ts'));
  } else if (basePath.endsWith('.d.ts')) {
    candidates.push(basePath);
  } else {
    candidates.push(`${basePath}.d.ts`, path.join(basePath, 'index.d.ts'));
  }
  for (const candidate of candidates) {
    if (await pathKind(candidate) === 'file') return candidate;
  }
  return null;
}

function isPackagedRuntimePath(repoPath) {
  return PACKAGE_RUNTIME_FILES.has(repoPath)
    || PACKAGE_RUNTIME_PREFIXES.some((prefix) => repoPath.startsWith(prefix));
}

async function scanRuntimeGraph(entrypoints) {
  const pending = [...entrypoints].map((entry) => path.join(ROOT_DIR, entry));
  const runtimeFiles = new Set();
  const resourceFiles = new Set(REQUIRED_RESOURCE_FILES);
  const issues = [];

  while (pending.length > 0) {
    const currentPath = pending.pop();
    const currentRepoPath = normalizePath(currentPath);
    if (runtimeFiles.has(currentRepoPath)) continue;
    if (await pathKind(currentPath) !== 'file') {
      issues.push(`runtime entry does not exist: ${currentRepoPath}`);
      continue;
    }
    if (!isPackagedRuntimePath(currentRepoPath)) {
      issues.push(`runtime dependency is outside the npm package roots: ${currentRepoPath}`);
      continue;
    }
    runtimeFiles.add(currentRepoPath);
    const source = await fs.readFile(currentPath, 'utf8');

    for (const specifier of collectModuleSpecifiers(source)) {
      if (!isLocalSpecifier(specifier)) continue;
      const resolved = await resolveRuntimeDependency(currentPath, specifier);
      if (!resolved) {
        issues.push(`${currentRepoPath} has unresolved local import: ${specifier}`);
        continue;
      }
      const resolvedRepoPath = normalizePath(resolved);
      if (resolvedRepoPath.endsWith('.js')) {
        pending.push(resolved);
      } else {
        resourceFiles.add(resolvedRepoPath);
      }
    }

    for (const specifier of collectRelativeFileLiterals(source)) {
      const resolved = await resolveRuntimeDependency(currentPath, specifier);
      if (!resolved) continue;
      const resolvedRepoPath = normalizePath(resolved);
      if (resolvedRepoPath.endsWith('.js')) {
        pending.push(resolved);
      } else {
        resourceFiles.add(resolvedRepoPath);
      }
    }
  }

  return { runtimeFiles, resourceFiles, issues };
}

async function scanTypeGraph(entrypoints) {
  const pending = [...entrypoints].map((entry) => path.join(ROOT_DIR, entry));
  const typeFiles = new Set();
  const issues = [];

  while (pending.length > 0) {
    const currentPath = pending.pop();
    const currentRepoPath = normalizePath(currentPath);
    if (typeFiles.has(currentRepoPath)) continue;
    if (await pathKind(currentPath) !== 'file') {
      issues.push(`type entry does not exist: ${currentRepoPath}`);
      continue;
    }
    typeFiles.add(currentRepoPath);
    const source = await fs.readFile(currentPath, 'utf8');
    for (const specifier of collectModuleSpecifiers(source)) {
      if (!isLocalSpecifier(specifier)) continue;
      const resolved = await resolveTypeDependency(currentPath, specifier);
      if (!resolved) {
        issues.push(`${currentRepoPath} has unresolved local type import: ${specifier}`);
        continue;
      }
      pending.push(resolved);
    }
  }

  return { typeFiles, issues };
}

async function collectPublicResourceFiles(resourceFiles) {
  const output = new Set(resourceFiles);
  for (const resourceDir of PUBLIC_RESOURCE_DIRS) {
    const absoluteDir = path.join(ROOT_DIR, resourceDir);
    for (const filePath of await walkFiles(absoluteDir, (entry) => entry.endsWith('.json'))) {
      output.add(normalizePath(filePath));
    }
  }

  const registryPath = path.join(ROOT_DIR, 'src/config/kernels/registry.json');
  const registry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
  output.add(normalizePath(registryPath));
  for (const operation of Object.values(registry.operations ?? {})) {
    for (const variant of Object.values(operation.variants ?? {})) {
      if (typeof variant?.wgsl === 'string') {
        output.add(`src/gpu/kernels/${variant.wgsl}`);
      }
    }
  }
  return output;
}

export async function buildPackageSourceClosure(packageJson) {
  const entrypoints = collectEntrypoints(packageJson);
  const runtime = await scanRuntimeGraph(entrypoints.runtime);
  const types = await scanTypeGraph(entrypoints.types);
  const packageResourceFiles = await collectPublicResourceFiles(runtime.resourceFiles);
  const allSourceCodeFiles = await walkFiles(
    SOURCE_DIR,
    (filePath) => filePath.endsWith('.js') || filePath.endsWith('.d.ts')
  );
  const allSourceJsonFiles = await walkFiles(SOURCE_DIR, (filePath) => filePath.endsWith('.json'));
  const ignoredSourceFiles = new Set();

  for (const filePath of allSourceCodeFiles) {
    const repoPath = normalizePath(filePath);
    const isRequired = repoPath.endsWith('.d.ts')
      ? types.typeFiles.has(repoPath)
      : runtime.runtimeFiles.has(repoPath);
    if (!isRequired) ignoredSourceFiles.add(repoPath.slice('src/'.length));
  }
  for (const filePath of allSourceJsonFiles) {
    const repoPath = normalizePath(filePath);
    if (!packageResourceFiles.has(repoPath)) {
      ignoredSourceFiles.add(repoPath.slice('src/'.length));
    }
  }

  return {
    runtimeFiles: runtime.runtimeFiles,
    typeFiles: types.typeFiles,
    packageResourceFiles,
    ignoredSourceFiles,
    issues: [...runtime.issues, ...types.issues],
  };
}

export function renderSourceNpmIgnore(closure) {
  const ignored = [...closure.ignoredSourceFiles]
    .filter((relativePath) => !ALWAYS_IGNORED_SOURCE_PREFIXES.some((prefix) => relativePath.startsWith(prefix)))
    .sort();
  return [
    '# Generated by tools/package-source-closure.js --write.',
    '# Runtime code is the closure of package exports/bins; declarations are the public type closure.',
    '',
    '**/*.md',
    ...ALWAYS_IGNORED_SOURCE_PREFIXES,
    ...ignored,
    '',
  ].join('\n');
}

export async function assertPackageSourceClosure(packageJson) {
  const closure = await buildPackageSourceClosure(packageJson);
  if (closure.issues.length > 0) {
    throw new Error(`npm package source closure has issues:\n${closure.issues.map((issue) => `- ${issue}`).join('\n')}`);
  }
  const expected = renderSourceNpmIgnore(closure);
  let actual = '';
  try {
    actual = await fs.readFile(SOURCE_IGNORE_PATH, 'utf8');
  } catch {
    throw new Error('src/.npmignore is missing; run npm run package:closure:sync.');
  }
  if (actual !== expected) {
    throw new Error('src/.npmignore is stale; run npm run package:closure:sync.');
  }
  return closure;
}

async function main() {
  const packageJson = JSON.parse(await fs.readFile(path.join(ROOT_DIR, 'package.json'), 'utf8'));
  const closure = await buildPackageSourceClosure(packageJson);
  if (closure.issues.length > 0) {
    throw new Error(closure.issues.join('\n'));
  }
  const content = renderSourceNpmIgnore(closure);
  if (process.argv.includes('--write')) {
    await fs.writeFile(SOURCE_IGNORE_PATH, content, 'utf8');
    console.log(
      `package source closure synced (${closure.runtimeFiles.size} runtime JS, `
      + `${closure.typeFiles.size} declarations, ${closure.packageResourceFiles.size} resources)`
    );
    return;
  }
  const current = await fs.readFile(SOURCE_IGNORE_PATH, 'utf8').catch(() => '');
  if (current !== content) {
    throw new Error('src/.npmignore is stale; run npm run package:closure:sync.');
  }
  console.log(
    `package source closure check passed (${closure.runtimeFiles.size} runtime JS, `
    + `${closure.typeFiles.size} declarations, ${closure.packageResourceFiles.size} resources)`
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
