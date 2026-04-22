#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const TEXT_FILES = Object.freeze([
  'src/config/schema/kernel-path.schema.d.ts',
  'src/config/schema/inference.schema.d.ts',
  'src/config/schema/index.d.ts',
  'src/config/kernel-path-loader.d.ts',
  'src/config/README.md',
  'docs/config.md',
  'docs/cli.md',
  'docs/style/general-style-guide.md',
  'docs/style/benchmark-style-guide.md',
  'docs/developer-guides/06-kernel-path-config.md',
  'docs/developer-guides/13-attention-variant.md',
]);

const TEXT_RULES = Object.freeze([
  {
    label: 'removed kernel-path asset directory',
    pattern: /src\/config\/kernel-paths/g,
  },
  {
    label: 'registered kernel-path ID wording',
    pattern: /registered kernel-path ID/gi,
  },
  {
    label: 'KernelPathRef string union',
    pattern: /KernelPathRef\s*=\s*string\s*\|/g,
  },
  {
    label: 'BuiltinKernelPathId export',
    pattern: /BuiltinKernelPathId/g,
  },
]);

const issues = [];

function toRepoPath(filePath) {
  return path.relative(ROOT, filePath).replace(/\\/g, '/');
}

function recordIssue(filePath, message) {
  issues.push(`${toRepoPath(filePath)}: ${message}`);
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function collectJsonFiles(dirPath) {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      files.push(...await collectJsonFiles(fullPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(fullPath);
    }
  }
  return files;
}

function scanRuntimeJsonValue(filePath, value, keyPath = []) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => scanRuntimeJsonValue(filePath, entry, keyPath.concat(String(index))));
    return;
  }
  if (!value || typeof value !== 'object') return;

  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    if (key === 'kernelPlan') {
      recordIssue(filePath, `${nextPath.join('.')} uses removed kernelPlan`);
    }
    if (key === 'kernelPath' && typeof child === 'string') {
      recordIssue(filePath, `${nextPath.join('.')} uses removed string kernel-path ID "${child}"`);
    }
    scanRuntimeJsonValue(filePath, child, nextPath);
  }
}

async function checkTextFiles() {
  for (const relPath of TEXT_FILES) {
    const filePath = path.join(ROOT, relPath);
    if (!await pathExists(filePath)) {
      recordIssue(filePath, 'expected contract file is missing');
      continue;
    }
    const text = await fs.readFile(filePath, 'utf8');
    for (const rule of TEXT_RULES) {
      const matches = text.match(rule.pattern);
      if (matches?.length) {
        recordIssue(filePath, `${rule.label} appears ${matches.length} time(s)`);
      }
    }
  }
}

async function checkRuntimeConfigs() {
  const runtimeRoot = path.join(ROOT, 'src/config/runtime');
  const files = await collectJsonFiles(runtimeRoot);
  for (const filePath of files) {
    const text = await fs.readFile(filePath, 'utf8');
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      recordIssue(filePath, `invalid JSON: ${error.message}`);
      continue;
    }
    scanRuntimeJsonValue(filePath, parsed);
  }
}

async function main() {
  await checkTextFiles();
  await checkRuntimeConfigs();

  if (issues.length > 0) {
    console.error('[config:single-source:check] found stale config contract drift:');
    for (const issue of issues) {
      console.error(`- ${issue}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log('[config:single-source:check] config contract single-source checks passed');
}

main().catch((error) => {
  console.error(`[config:single-source:check] ${error.stack || error.message}`);
  process.exitCode = 1;
});
