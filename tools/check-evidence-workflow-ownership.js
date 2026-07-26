#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const POLICY_PATH = path.join(
  REPO_ROOT,
  'src',
  'config',
  'evidence',
  'workflow-ownership-policy.json'
);
const ACTIVE_SCAN_ROOTS = ['src', 'tools', 'tests'];
const EXCLUDED_SCAN_PATHS = new Set([
  'src/config/evidence/workflow-ownership-policy.json',
  'tools/check-evidence-workflow-ownership.js',
]);

async function pathExists(relativePath) {
  try {
    await fs.stat(path.join(REPO_ROOT, relativePath));
    return true;
  } catch {
    return false;
  }
}

async function listFiles(relativeRoot) {
  const absoluteRoot = path.join(REPO_ROOT, relativeRoot);
  const entries = await fs.readdir(absoluteRoot, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const relativePath = path.posix.join(relativeRoot, entry.name);
    if (entry.isDirectory()) {
      files.push(...await listFiles(relativePath));
    } else if (entry.isFile() && /\.(?:js|d\.ts|json)$/.test(entry.name)) {
      files.push(relativePath);
    }
  }
  return files;
}

export async function buildEvidenceWorkflowOwnershipReport() {
  const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));
  const errors = [];
  if (policy.schema !== 'doppler.evidence-workflow-ownership-policy/v1') {
    errors.push('workflow ownership policy schema is invalid');
  }
  const seenJobs = new Set();
  for (const entry of policy.jobs ?? []) {
    if (seenJobs.has(entry.job)) {
      errors.push(`duplicate workflow job: ${entry.job}`);
    }
    seenJobs.add(entry.job);
    if (!await pathExists(entry.canonicalOwner)) {
      errors.push(`${entry.job}: canonical owner does not exist: ${entry.canonicalOwner}`);
    }
  }
  for (const forbiddenPath of policy.forbiddenPaths ?? []) {
    if (await pathExists(forbiddenPath)) {
      errors.push(`retired workflow path still exists: ${forbiddenPath}`);
    }
  }
  const activeFiles = (
    await Promise.all(ACTIVE_SCAN_ROOTS.map(listFiles))
  ).flat().filter((filePath) => !EXCLUDED_SCAN_PATHS.has(filePath));
  for (const filePath of activeFiles) {
    const source = await fs.readFile(path.join(REPO_ROOT, filePath), 'utf8');
    for (const symbol of policy.forbiddenActiveSymbols ?? []) {
      if (source.includes(symbol)) {
        errors.push(`retired workflow symbol "${symbol}" remains in ${filePath}`);
      }
    }
  }
  return {
    schema: 'doppler.evidence-workflow-ownership-check/v1',
    ok: errors.length === 0,
    policyPath: path.relative(REPO_ROOT, POLICY_PATH),
    jobs: seenJobs.size,
    scannedFiles: activeFiles.length,
    errors,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((argument) => argument !== '--json');
  if (unsupported.length > 0) {
    throw new Error(`Unknown argument: ${unsupported[0]}`);
  }
  const report = await buildEvidenceWorkflowOwnershipReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(`evidence-workflow-ownership: ok (${report.jobs} canonical jobs)`);
  } else {
    for (const error of report.errors) {
      console.error(`evidence-workflow-ownership: ${error}`);
    }
  }
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
