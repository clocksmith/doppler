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
const PACKAGE_PATH = path.join(REPO_ROOT, 'package.json');
const ACTIVE_SCAN_ROOTS = ['src', 'tools', 'tests'];
const EXCLUDED_SCAN_PATHS = new Set([
  'src/config/evidence/workflow-ownership-policy.json',
  'tools/check-evidence-workflow-ownership.js',
]);

async function pathExists(relativePath) {
  if (typeof relativePath !== 'string' || !relativePath.trim()) return false;
  try {
    await fs.stat(path.join(REPO_ROOT, relativePath));
    return true;
  } catch {
    return false;
  }
}

function isRepoRelativePath(value) {
  return typeof value === 'string'
    && Boolean(value.trim())
    && !path.isAbsolute(value)
    && !value.includes('\\')
    && !value.split('/').includes('..');
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
  const [policy, packageJson] = await Promise.all([
    fs.readFile(POLICY_PATH, 'utf8').then(JSON.parse),
    fs.readFile(PACKAGE_PATH, 'utf8').then(JSON.parse),
  ]);
  const errors = [];
  if (policy.schema !== 'doppler.evidence-workflow-ownership-policy/v1') {
    errors.push('workflow ownership policy schema is invalid');
  }
  const seenJobs = new Set();
  const seenOwners = new Set();
  const seenCommands = new Set();
  for (const entry of policy.jobs ?? []) {
    const fields = ['job', 'canonicalOwner', 'canonicalCommand', 'adapters'];
    if (entry == null || typeof entry !== 'object' || Array.isArray(entry)) {
      errors.push('workflow job must be an object');
      continue;
    }
    const actualFields = Object.keys(entry);
    for (const field of fields) {
      if (!Object.hasOwn(entry, field)) errors.push(`${entry.job || '<missing-job>'}: ${field} is required`);
    }
    for (const field of actualFields) {
      if (!fields.includes(field)) errors.push(`${entry.job || '<missing-job>'}: ${field} is not supported`);
    }
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(entry.job || '')) {
      errors.push(`workflow job id must be lowercase kebab-case: ${String(entry.job)}`);
    }
    if (seenJobs.has(entry.job)) {
      errors.push(`duplicate workflow job: ${entry.job}`);
    }
    seenJobs.add(entry.job);
    if (seenOwners.has(entry.canonicalOwner)) {
      errors.push(`duplicate canonical workflow owner: ${entry.canonicalOwner}`);
    }
    seenOwners.add(entry.canonicalOwner);
    if (!isRepoRelativePath(entry.canonicalOwner)) {
      errors.push(`${entry.job}: canonical owner must be a repo-relative path`);
    } else if (!await pathExists(entry.canonicalOwner)) {
      errors.push(`${entry.job}: canonical owner does not exist: ${entry.canonicalOwner}`);
    }
    if (entry.canonicalCommand !== null && typeof entry.canonicalCommand !== 'string') {
      errors.push(`${entry.job}: canonicalCommand must be a non-empty string or null`);
    } else if (typeof entry.canonicalCommand === 'string') {
      const command = entry.canonicalCommand.trim();
      if (!command) {
        errors.push(`${entry.job}: canonicalCommand must be a non-empty string or null`);
      } else if (seenCommands.has(command)) {
        errors.push(`duplicate canonical workflow command: ${command}`);
      } else {
        seenCommands.add(command);
      }
      const match = /^npm run ([a-z0-9:-]+)$/.exec(command);
      if (command.startsWith('npm run ') && (!match || !packageJson.scripts?.[match[1]])) {
        errors.push(`${entry.job}: canonical npm command is not declared in package.json: ${command}`);
      }
    }
    if (!Array.isArray(entry.adapters)) {
      errors.push(`${entry.job}: adapters must be an array`);
    } else if (new Set(entry.adapters).size !== entry.adapters.length) {
      errors.push(`${entry.job}: adapters must not contain duplicates`);
    } else if (entry.adapters.some((adapter) => typeof adapter !== 'string' || !adapter.trim())) {
      errors.push(`${entry.job}: adapters must contain non-empty strings`);
    } else {
      for (const adapter of entry.adapters) {
        if (/^(?:src|tools|tests)\//.test(adapter) && !await pathExists(adapter)) {
          errors.push(`${entry.job}: adapter path does not exist: ${adapter}`);
        }
      }
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
