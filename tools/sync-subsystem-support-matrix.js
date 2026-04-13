#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import {
  SUPPORT_TIERS,
  validateSupportTierRegistry,
} from '../src/config/schema/support-tiers.schema.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const REGISTRY_PATH = path.join(REPO_ROOT, 'src/config/support-tiers/subsystems.json');
const DEFAULT_OUTPUT_PATH = path.join(REPO_ROOT, 'docs/subsystem-support-matrix.md');
const PACKAGE_JSON_PATH = path.join(REPO_ROOT, 'package.json');

const TIER_ORDER = new Map(SUPPORT_TIERS.map((entry, index) => [entry, index]));
const CLAIM_ORDER = new Map([
  ['primary', 0],
  ['secondary', 1],
  ['none', 2],
]);

export function parseArgs(argv) {
  const args = {
    check: false,
    outputPath: DEFAULT_OUTPUT_PATH,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const entry = argv[i];
    const nextValue = () => {
      const candidate = argv[i + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${entry}`);
      }
      i += 1;
      return String(candidate).trim();
    };
    if (entry === '--check') {
      args.check = true;
      continue;
    }
    if (entry === '--output') {
      const candidate = nextValue();
      if (!candidate) {
        throw new Error('Missing value for --output');
      }
      args.outputPath = path.resolve(REPO_ROOT, candidate);
      continue;
    }
    throw new Error(`Unknown argument: ${entry}`);
  }
  return args;
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

function relativePath(filePath) {
  return path.relative(REPO_ROOT, filePath).replace(/\\/g, '/');
}

function markdownRelativePath(fromFile, targetFile) {
  return path.relative(path.dirname(fromFile), targetFile).replace(/\\/g, '/');
}

async function assertPathExists(repoRelativePath, label) {
  const fullPath = path.join(REPO_ROOT, repoRelativePath);
  try {
    await fs.access(fullPath);
  } catch {
    throw new Error(`support tiers: ${label} path does not exist: ${repoRelativePath}`);
  }
}

async function validateRegistryAgainstRepo(registry) {
  const packageJson = await readJson(PACKAGE_JSON_PATH);
  const packageExports = packageJson && typeof packageJson.exports === 'object' ? packageJson.exports : {};
  const packageBins = packageJson && typeof packageJson.bin === 'object' ? packageJson.bin : {};

  for (const entry of registry.subsystems) {
    if (entry.exported && !Object.prototype.hasOwnProperty.call(packageExports, entry.packageExport)) {
      throw new Error(
        `support tiers: exported subsystem "${entry.id}" references missing package export "${entry.packageExport}".`
      );
    }
    if (entry.command != null && !Object.prototype.hasOwnProperty.call(packageBins, entry.command)) {
      throw new Error(
        `support tiers: subsystem "${entry.id}" references missing package bin "${entry.command}".`
      );
    }
    for (const docPath of entry.docs) {
      await assertPathExists(docPath, `${entry.id}.docs`);
    }
    for (const entrypoint of entry.entrypoints) {
      await assertPathExists(entrypoint, `${entry.id}.entrypoints`);
    }
  }
}

function renderLink(outputPath, repoRelativePath) {
  const absoluteTarget = path.join(REPO_ROOT, repoRelativePath);
  const relativeTarget = markdownRelativePath(outputPath, absoluteTarget);
  return `[${repoRelativePath}](${relativeTarget})`;
}

function renderAnchorLinks(entry, outputPath, limit = 3) {
  const anchors = [];
  for (const docPath of entry.docs) {
    anchors.push(renderLink(outputPath, docPath));
  }
  for (const entrypoint of entry.entrypoints) {
    anchors.push(renderLink(outputPath, entrypoint));
  }
  return anchors.slice(0, limit).join(', ');
}

function renderBoolean(value) {
  return value ? 'yes' : 'no';
}

function sortSubsystems(entries) {
  return [...entries].sort((left, right) => {
    const tierDelta = (TIER_ORDER.get(left.tier) ?? 999) - (TIER_ORDER.get(right.tier) ?? 999);
    if (tierDelta !== 0) return tierDelta;
    const claimDelta = (CLAIM_ORDER.get(left.claimVisibility) ?? 999) - (CLAIM_ORDER.get(right.claimVisibility) ?? 999);
    if (claimDelta !== 0) return claimDelta;
    return left.label.localeCompare(right.label);
  });
}

function renderTierSection(outputPath, tier, entries) {
  const title = tier === 'tier1' ? 'Tier 1 Surfaces' : tier === 'experimental' ? 'Experimental Surfaces' : 'Internal-only Surfaces';
  const lines = [];
  lines.push(`## ${title}`);
  lines.push('');
  lines.push('| Subsystem | Scope | User-facing | Demo default | Exported | Anchors | Notes |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- |');
  for (const entry of sortSubsystems(entries).filter((candidate) => candidate.tier === tier)) {
    lines.push(
      `| ${entry.label} | \`${entry.scope}\` | ${renderBoolean(entry.userFacing)} | ${renderBoolean(entry.demoDefault)} | ${renderBoolean(entry.exported)} | ${renderAnchorLinks(entry, outputPath)} | ${entry.notes} |`
    );
  }
  lines.push('');
  return lines;
}

function renderReadmeClaims(outputPath, entries) {
  const claims = sortSubsystems(entries.filter((entry) => entry.claimVisibility !== 'none'));
  const lines = [];
  lines.push('## README-Facing Claims');
  lines.push('');
  lines.push('| Claim | Visibility | Tier | Anchors | Notes |');
  lines.push('| --- | --- | --- | --- | --- |');
  for (const entry of claims) {
    lines.push(
      `| ${entry.label} | \`${entry.claimVisibility}\` | \`${entry.tier}\` | ${renderAnchorLinks(entry, outputPath)} | ${entry.notes} |`
    );
  }
  lines.push('');
  return lines;
}

export function renderSubsystemSupportMatrix(registry, outputPath) {
  const counts = {
    tier1: registry.subsystems.filter((entry) => entry.tier === 'tier1').length,
    experimental: registry.subsystems.filter((entry) => entry.tier === 'experimental').length,
    internalOnly: registry.subsystems.filter((entry) => entry.tier === 'internal-only').length,
  };
  const lines = [];
  lines.push('# Subsystem Support Matrix');
  lines.push('');
  lines.push('Auto-generated from `src/config/support-tiers/subsystems.json`.');
  lines.push('Run `npm run support:subsystems:sync` after editing the subsystem-tier registry.');
  lines.push('');
  lines.push(`Updated at: ${registry.updatedAtUtc.slice(0, 10)}`);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Tier 1 subsystems: ${counts.tier1}`);
  lines.push(`- Experimental subsystems: ${counts.experimental}`);
  lines.push(`- Internal-only subsystems: ${counts.internalOnly}`);
  lines.push('');
  lines.push('## Tier Meanings');
  lines.push('');
  lines.push('- `tier1`: current public support contract and claimable mainline behavior.');
  lines.push('- `experimental`: checked-in and sometimes exported, but not part of the canonical quickstart or demo-default proof path.');
  lines.push('- `internal-only`: repo machinery that should not be treated as part of the public product contract.');
  lines.push('');
  lines.push(...renderReadmeClaims(outputPath, registry.subsystems));
  lines.push(...renderTierSection(outputPath, 'tier1', registry.subsystems));
  lines.push(...renderTierSection(outputPath, 'experimental', registry.subsystems));
  lines.push(...renderTierSection(outputPath, 'internal-only', registry.subsystems));
  return `${lines.join('\n').trim()}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const registry = validateSupportTierRegistry(await readJson(REGISTRY_PATH));
  await validateRegistryAgainstRepo(registry);
  const rendered = renderSubsystemSupportMatrix(registry, args.outputPath);
  if (args.check) {
    let existing;
    try {
      existing = await fs.readFile(args.outputPath, 'utf8');
    } catch {
      throw new Error(`Missing ${relativePath(args.outputPath)}. Run npm run support:subsystems:sync`);
    }
    if (existing !== rendered) {
      throw new Error(`${relativePath(args.outputPath)} is stale.\nRun npm run support:subsystems:sync`);
    }
    console.log(`[support:subsystems:check] ${relativePath(args.outputPath)} is up to date`);
    return;
  }
  await fs.writeFile(args.outputPath, rendered, 'utf8');
  console.log(`[support:subsystems:sync] wrote ${relativePath(args.outputPath)}`);
}

if (process.argv[1] === __filename) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
