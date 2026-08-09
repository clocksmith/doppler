#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const policyPath = path.join(repoRoot, 'tools/policies/source-architecture-policy.json');
const importPattern = /\b(?:import|export)\s+(?:[^'\"]*?\sfrom\s*)?['\"]([^'\"]+)['\"]/g;
const dynamicImportPattern = /\bimport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)/g;
const facadeImplementationPattern = /^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\b/m;
const governedExtensions = new Set(['.js', '.d.ts', '.wgsl']);

function toPosix(value) {
  return value.split(path.sep).join('/');
}

function countLines(source) {
  if (source.length === 0) return 0;
  const lines = source.split(/\r?\n/);
  return lines.at(-1) === '' ? lines.length - 1 : lines.length;
}

async function walk(directory) {
  const files = [];
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await walk(entryPath));
    else if (entry.isFile()) files.push(entryPath);
  }
  return files;
}

function collectSpecifiers(source) {
  const values = [];
  for (const pattern of [importPattern, dynamicImportPattern]) {
    pattern.lastIndex = 0;
    for (;;) {
      const match = pattern.exec(source);
      if (!match) break;
      values.push(match[1]);
    }
  }
  return values;
}

function ownerFor(sourceRoot, targetPath) {
  const relative = path.relative(sourceRoot, targetPath);
  if (relative.startsWith('..') || path.isAbsolute(relative)) return null;
  return relative.split(path.sep)[0];
}

function exceptionKey(from, toOwner) {
  return `${from}->${toOwner}`;
}

async function main() {
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const sourceRoot = path.join(repoRoot, policy.sourceRoot);
  const errors = [];
  const actualOwners = (await fs.readdir(sourceRoot, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  const expectedOwners = Object.keys(policy.topLevelOwners).sort();
  if (JSON.stringify(actualOwners) !== JSON.stringify(expectedOwners)) {
    errors.push(`top-level source owners differ: expected=${expectedOwners.join(',')} actual=${actualOwners.join(',')}`);
  }

  const exceptions = new Map();
  for (const entry of policy.dependencyExceptions) {
    if (!entry.reason || !entry.extractTo) {
      errors.push(`dependency exception lacks reason or extractTo: ${JSON.stringify(entry)}`);
      continue;
    }
    const key = exceptionKey(entry.from, entry.toOwner);
    if (exceptions.has(key)) errors.push(`duplicate dependency exception: ${key}`);
    exceptions.set(key, { ...entry, used: false });
  }

  const legacyOversize = new Map(Object.entries(policy.legacyOversize));
  const observedLegacy = new Set();
  const files = await walk(sourceRoot);
  for (const filePath of files) {
    const relative = toPosix(path.relative(sourceRoot, filePath));
    const source = await fs.readFile(filePath, 'utf8');
    const extension = relative.endsWith('.d.ts') ? '.d.ts' : path.extname(relative);
    if (governedExtensions.has(extension)) {
      const lineCount = countLines(source);
      const legacy = legacyOversize.get(relative);
      if (legacy) {
        observedLegacy.add(relative);
        if (!Array.isArray(legacy.extractTo) || legacy.extractTo.length === 0) {
          errors.push(`${relative}: legacy oversize entry lacks extraction boundaries`);
        }
        if (lineCount > legacy.maxLines) {
          errors.push(`${relative}: grew from governed ceiling ${legacy.maxLines} to ${lineCount} lines`);
        }
        if (lineCount <= policy.lineLimit) {
          errors.push(`${relative}: now satisfies ${policy.lineLimit}; remove stale legacyOversize entry`);
        }
      } else if (lineCount > policy.lineLimit) {
        errors.push(`${relative}: ${lineCount} lines exceeds blocking limit ${policy.lineLimit}`);
      }
    }

    if (path.extname(relative) !== '.js') continue;
    const fromOwner = relative.split('/')[0];
    const allowedOwners = policy.restrictedDependencies[fromOwner];
    if (!allowedOwners) continue;
    for (const specifier of collectSpecifiers(source)) {
      if (!specifier.startsWith('.')) continue;
      const targetPath = path.resolve(path.dirname(filePath), specifier);
      const toOwner = ownerFor(sourceRoot, targetPath);
      if (!toOwner || allowedOwners.includes(toOwner)) continue;
      const key = exceptionKey(relative, toOwner);
      const exception = exceptions.get(key);
      if (exception) {
        exception.used = true;
        continue;
      }
      errors.push(`${relative}: dependency on ${toOwner} violates ${fromOwner} ownership boundary`);
    }
  }

  for (const relative of legacyOversize.keys()) {
    if (!observedLegacy.has(relative)) errors.push(`legacyOversize entry is stale or missing: ${relative}`);
  }
  for (const [key, entry] of exceptions) {
    if (!entry.used) errors.push(`dependency exception is stale: ${key}`);
  }
  for (const relative of policy.facades) {
    const facadePath = path.join(sourceRoot, relative);
    const source = await fs.readFile(facadePath, 'utf8').catch(() => null);
    if (source === null) {
      errors.push(`declared facade is missing: ${relative}`);
      continue;
    }
    if (facadeImplementationPattern.test(source)) {
      errors.push(`${relative}: facade contains an implementation declaration`);
    }
  }

  if (errors.length > 0) {
    console.error('source architecture check failed:');
    for (const error of errors) console.error(`- ${error}`);
    process.exitCode = 1;
    return;
  }
  console.log(`source architecture check passed: ${actualOwners.length} owners, ${files.length} files, ${legacyOversize.size} governed oversize files`);
}

await main();
