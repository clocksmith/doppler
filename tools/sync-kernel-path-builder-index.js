#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { getKernelPath } from '../src/config/kernel-path-loader.js';
import { buildKernelPathBuilderArtifacts } from '../src/tooling/kernel-path-builder/index.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CONFIG_ROOT = path.join(REPO_ROOT, 'src', 'config', 'conversion');
const DEFAULT_MANIFEST_ROOT = path.join(REPO_ROOT, 'models');
const DEFAULT_REGISTRY_FILE = path.join(REPO_ROOT, 'src', 'config', 'kernel-paths', 'registry.json');
const DEFAULT_INDEX_OUTPUT_FILE = path.join(REPO_ROOT, 'demo', 'data', 'kernel-path-builder-index.json');
const DEFAULT_PROPOSALS_OUTPUT_FILE = path.join(REPO_ROOT, 'demo', 'data', 'kernel-path-builder-proposals.json');
const DEFAULT_REPORT_OUTPUT_FILE = path.join(REPO_ROOT, 'demo', 'data', 'kernel-path-builder-report.md');

function parseArgs(argv) {
  const args = {
    check: false,
    configRoot: DEFAULT_CONFIG_ROOT,
    manifestRoot: DEFAULT_MANIFEST_ROOT,
    registryFile: DEFAULT_REGISTRY_FILE,
    outputFile: DEFAULT_INDEX_OUTPUT_FILE,
    indexOutputFile: DEFAULT_INDEX_OUTPUT_FILE,
    proposalsOutputFile: DEFAULT_PROPOSALS_OUTPUT_FILE,
    reportOutputFile: DEFAULT_REPORT_OUTPUT_FILE,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const nextPath = () => {
      const candidate = argv[index + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${arg}`);
      }
      index += 1;
      return path.resolve(REPO_ROOT, String(candidate).trim());
    };
    if (arg === '--check') {
      args.check = true;
      continue;
    }
    if (arg === '--config-root') {
      args.configRoot = nextPath();
      continue;
    }
    if (arg === '--manifest-root') {
      args.manifestRoot = nextPath();
      continue;
    }
    if (arg === '--registry-file') {
      args.registryFile = nextPath();
      continue;
    }
    if (arg === '--output-file') {
      args.outputFile = nextPath();
      args.indexOutputFile = args.outputFile;
      continue;
    }
    if (arg === '--index-output-file') {
      args.indexOutputFile = nextPath();
      continue;
    }
    if (arg === '--proposals-output-file') {
      args.proposalsOutputFile = nextPath();
      continue;
    }
    if (arg === '--report-output-file') {
      args.reportOutputFile = nextPath();
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function normalizeRepoPath(filePath) {
  return path.relative(REPO_ROOT, filePath).replace(/\\/g, '/');
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

async function collectFiles(rootDir, predicate) {
  if (!rootDir || !(await pathExists(rootDir))) {
    return [];
  }
  const files = [];
  async function walk(currentDir) {
    const entries = await fs.readdir(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const absolutePath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        await walk(absolutePath);
        continue;
      }
      if (entry.isFile() && predicate(entry.name, absolutePath)) {
        files.push(absolutePath);
      }
    }
  }
  await walk(rootDir);
  files.sort((left, right) => left.localeCompare(right));
  return files;
}

async function loadConfigEntries(configRoot) {
  const files = await collectFiles(configRoot, (name) => name.endsWith('.json'));
  const entries = [];
  for (const filePath of files) {
    entries.push({
      configPath: normalizeRepoPath(filePath),
      rawConfig: await readJson(filePath),
    });
  }
  return entries;
}

async function loadManifestEntries(manifestRoot) {
  const files = await collectFiles(manifestRoot, (name) => name === 'manifest.json');
  const entries = [];
  for (const filePath of files) {
    entries.push({
      manifestPath: normalizeRepoPath(filePath),
      manifest: await readJson(filePath),
    });
  }
  return entries;
}

async function loadRegistryEntries(registryFile) {
  const registryPayload = await readJson(registryFile);
  const rawEntries = Array.isArray(registryPayload?.entries) ? registryPayload.entries : [];
  return rawEntries.map((entry) => ({
    id: String(entry?.id ?? '').trim(),
    status: String(entry?.status ?? 'canonical').trim() || 'canonical',
    statusReason: String(entry?.statusReason ?? '').trim(),
    notes: String(entry?.notes ?? '').trim(),
    path: getKernelPath(String(entry?.id ?? '').trim()),
  }));
}

function ensureTrailingNewline(value) {
  return String(value ?? '').endsWith('\n')
    ? String(value ?? '')
    : `${String(value ?? '')}\n`;
}

function resolveOutputFiles(options = {}) {
  return {
    index: path.resolve(REPO_ROOT, options.indexOutputFile ?? options.outputFile ?? DEFAULT_INDEX_OUTPUT_FILE),
    proposals: path.resolve(REPO_ROOT, options.proposalsOutputFile ?? DEFAULT_PROPOSALS_OUTPUT_FILE),
    report: path.resolve(REPO_ROOT, options.reportOutputFile ?? DEFAULT_REPORT_OUTPUT_FILE),
  };
}

function serializeKernelPathBuilderArtifacts(artifacts) {
  return {
    index: ensureTrailingNewline(JSON.stringify(artifacts.index, null, 2)),
    proposals: ensureTrailingNewline(JSON.stringify(artifacts.proposals, null, 2)),
    report: ensureTrailingNewline(artifacts.reportMarkdown),
  };
}

async function buildKernelPathBuilderInput(options = {}) {
  const configEntries = await loadConfigEntries(options.configRoot ?? DEFAULT_CONFIG_ROOT);
  const manifestEntries = await loadManifestEntries(options.manifestRoot ?? DEFAULT_MANIFEST_ROOT);
  const registryEntries = await loadRegistryEntries(options.registryFile ?? DEFAULT_REGISTRY_FILE);
  return {
    configEntries,
    manifestEntries,
    registryEntries,
  };
}

export async function buildKernelPathBuilderArtifactsPayload(options = {}) {
  const input = await buildKernelPathBuilderInput(options);
  return buildKernelPathBuilderArtifacts(input);
}

export async function buildKernelPathBuilderIndexPayload(options = {}) {
  const artifacts = await buildKernelPathBuilderArtifactsPayload(options);
  return artifacts.index;
}

export async function buildKernelPathBuilderContractArtifact(options = {}) {
  const artifacts = await buildKernelPathBuilderArtifactsPayload(options);
  const expectedTexts = serializeKernelPathBuilderArtifacts(artifacts);
  const outputFiles = resolveOutputFiles(options);
  const checks = [];
  const errors = [];

  for (const [artifactId, filePath] of Object.entries(outputFiles)) {
    const exists = await pathExists(filePath);
    let matches = false;
    if (exists) {
      const currentText = await fs.readFile(filePath, 'utf8');
      matches = currentText === expectedTexts[artifactId];
    }
    checks.push({
      id: `kernelPathBuilder.${artifactId}.fresh`,
      ok: exists && matches,
    });
    if (!exists) {
      errors.push(
        `[KernelPathBuilder] missing generated artifact ${normalizeRepoPath(filePath)}. ` +
        'Run npm run kernel-path:index:sync.'
      );
      continue;
    }
    if (!matches) {
      errors.push(
        `[KernelPathBuilder] generated artifact is out of date at ${normalizeRepoPath(filePath)}. ` +
        'Run npm run kernel-path:index:sync.'
      );
    }
  }

  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: errors.length === 0,
    checks,
    errors,
    stats: {
      models: artifacts.index?.stats?.models ?? 0,
      proposals: artifacts.proposals?.stats?.proposals ?? 0,
      verifiedProposals: artifacts.proposals?.stats?.verified ?? 0,
      manifestSources: artifacts.index?.stats?.manifestSources ?? 0,
      outputFiles: Object.keys(outputFiles).length,
    },
  };
}

async function writeKernelPathBuilderArtifacts(outputFiles, artifactTexts) {
  for (const filePath of Object.values(outputFiles)) {
    await fs.mkdir(path.dirname(filePath), { recursive: true });
  }
  await fs.writeFile(outputFiles.index, artifactTexts.index, 'utf8');
  await fs.writeFile(outputFiles.proposals, artifactTexts.proposals, 'utf8');
  await fs.writeFile(outputFiles.report, artifactTexts.report, 'utf8');
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.check) {
    const artifact = await buildKernelPathBuilderContractArtifact(args);
    if (!artifact.ok) {
      throw new Error(artifact.errors[0] ?? 'kernel-path builder artifacts are out of date.');
    }
    console.log(
      `[kernel-path-builder] up to date ` +
      `(models=${artifact.stats.models} proposals=${artifact.stats.proposals} verified=${artifact.stats.verifiedProposals})`
    );
    return;
  }

  const artifacts = await buildKernelPathBuilderArtifactsPayload(args);
  const outputFiles = resolveOutputFiles(args);
  const artifactTexts = serializeKernelPathBuilderArtifacts(artifacts);
  await writeKernelPathBuilderArtifacts(outputFiles, artifactTexts);
  console.log(
    `[kernel-path-builder] wrote ${normalizeRepoPath(outputFiles.index)}, ` +
    `${normalizeRepoPath(outputFiles.proposals)}, ${normalizeRepoPath(outputFiles.report)} ` +
    `(models=${artifacts.index.models.length} proposals=${artifacts.proposals.stats?.proposals ?? 0})`
  );
}

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  main().catch((error) => {
    console.error(`[kernel-path-builder] ${error.message}`);
    process.exit(1);
  });
}
