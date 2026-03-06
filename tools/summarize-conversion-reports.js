#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { validateConversionReport } from '../src/config/schema/conversion-report.schema.js';

function parseArgs(argv) {
  const args = {
    root: process.env.DOPPLER_REPORTS_DIR || path.resolve(process.cwd(), 'reports'),
    json: false,
    limit: 20,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--root') {
      args.root = path.resolve(argv[index + 1] ?? '');
      index += 1;
      continue;
    }
    if (arg === '--json') {
      args.json = true;
      continue;
    }
    if (arg === '--limit') {
      args.limit = Number(argv[index + 1] ?? args.limit);
      index += 1;
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      args.help = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!Number.isInteger(args.limit) || args.limit < 1) {
    throw new Error('--limit must be a positive integer.');
  }
  return args;
}

function usage() {
  return [
    'Usage:',
    '  node tools/summarize-conversion-reports.js [--root <reports-dir>] [--limit <n>] [--json]',
  ].join('\n');
}

export async function collectJsonFiles(rootDir) {
  const out = [];
  async function walk(currentDir) {
    let entries;
    try {
      entries = await fs.readdir(currentDir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const absolute = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        await walk(absolute);
        continue;
      }
      if (entry.isFile() && entry.name.endsWith('.json')) {
        out.push(absolute);
      }
    }
  }
  await walk(rootDir);
  return out;
}

export async function loadConversionReports(rootDir) {
  const files = await collectJsonFiles(rootDir);
  const reports = [];
  for (const filePath of files) {
    try {
      const parsed = JSON.parse(await fs.readFile(filePath, 'utf8'));
      validateConversionReport(parsed);
      reports.push({ path: filePath, report: parsed });
    } catch {
      continue;
    }
  }
  reports.sort((left, right) => String(right.report.timestamp).localeCompare(String(left.report.timestamp)));
  return reports;
}

function firstError(artifact) {
  return Array.isArray(artifact?.errors) && artifact.errors.length > 0
    ? String(artifact.errors[0])
    : null;
}

function artifactStatus(artifact) {
  return artifact?.ok === true
    ? 'pass'
    : artifact
      ? 'fail'
      : 'n/a';
}

export function buildSummary(entries, rootDir, limit) {
  return {
    schemaVersion: 1,
    source: 'doppler',
    root: rootDir,
    totalReports: entries.length,
    contractPass: entries.filter((entry) => entry.report.executionContractArtifact?.ok === true).length,
    contractFail: entries.filter((entry) => entry.report.executionContractArtifact?.ok === false).length,
    graphPass: entries.filter((entry) => entry.report.executionV0GraphContractArtifact?.ok === true).length,
    graphFail: entries.filter((entry) => entry.report.executionV0GraphContractArtifact?.ok === false).length,
    layerPatternPass: entries.filter((entry) => entry.report.layerPatternContractArtifact?.ok === true).length,
    layerPatternFail: entries.filter((entry) => entry.report.layerPatternContractArtifact?.ok === false).length,
    requiredInferencePass: entries.filter((entry) => entry.report.requiredInferenceFieldsArtifact?.ok === true).length,
    requiredInferenceFail: entries.filter((entry) => entry.report.requiredInferenceFieldsArtifact?.ok === false).length,
    recent: entries.slice(0, limit).map(({ path: filePath, report }) => ({
      modelId: report.modelId,
      timestamp: report.timestamp,
      presetId: report.result.presetId,
      modelType: report.result.modelType,
      contractOk: report.executionContractArtifact?.ok === true,
      graphOk: report.executionV0GraphContractArtifact?.ok === true,
      layerPatternOk: report.layerPatternContractArtifact?.ok === true,
      requiredInferenceOk: report.requiredInferenceFieldsArtifact?.ok === true,
      contractStatus: artifactStatus(report.executionContractArtifact),
      graphStatus: artifactStatus(report.executionV0GraphContractArtifact),
      layerPatternStatus: artifactStatus(report.layerPatternContractArtifact),
      requiredInferenceStatus: artifactStatus(report.requiredInferenceFieldsArtifact),
      layout: report.executionContractArtifact?.session?.layout ?? null,
      checks: Array.isArray(report.executionContractArtifact?.checks)
        ? report.executionContractArtifact.checks.length
        : 0,
      graphChecks: Array.isArray(report.executionV0GraphContractArtifact?.checks)
        ? report.executionV0GraphContractArtifact.checks.length
        : 0,
      contractFirstError: firstError(report.executionContractArtifact),
      graphFirstError: firstError(report.executionV0GraphContractArtifact),
      layerPatternFirstError: firstError(report.layerPatternContractArtifact),
      requiredInferenceFirstError: firstError(report.requiredInferenceFieldsArtifact),
      path: path.relative(process.cwd(), filePath),
    })),
  };
}

function printHuman(summary) {
  console.log(
    `[conversion-reports] total=${summary.totalReports} ` +
    `contractPass=${summary.contractPass} contractFail=${summary.contractFail} ` +
    `graphPass=${summary.graphPass} graphFail=${summary.graphFail} ` +
    `layerPatternPass=${summary.layerPatternPass} layerPatternFail=${summary.layerPatternFail} ` +
    `requiredInferencePass=${summary.requiredInferencePass} requiredInferenceFail=${summary.requiredInferenceFail}`
  );
  for (const entry of summary.recent) {
    console.log(
      `[conversion-reports] ${entry.timestamp} model=${entry.modelId} ` +
      `preset=${entry.presetId ?? 'n/a'} type=${entry.modelType ?? 'n/a'} ` +
      `contract=${entry.contractStatus} ` +
      `graph=${entry.graphStatus} ` +
      `layerPattern=${entry.layerPatternStatus} ` +
      `requiredInference=${entry.requiredInferenceStatus} ` +
      `layout=${entry.layout ?? 'n/a'} ` +
      `path=${entry.path}`
    );
    if (entry.graphOk === false && entry.graphFirstError) {
      console.log(`[conversion-reports] graph_error=${JSON.stringify(entry.graphFirstError)}`);
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const reports = await loadConversionReports(args.root);
  const summary = buildSummary(reports, args.root, args.limit);
  if (args.json) {
    console.log(JSON.stringify(summary, null, 2));
    return;
  }
  printHuman(summary);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
