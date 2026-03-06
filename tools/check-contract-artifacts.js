#!/usr/bin/env node

import { buildMergeContractArtifact } from '../src/config/merge-contract-check.js';
import { getKernelPathContractArtifact } from '../src/config/kernel-path-loader.js';
import { buildQuantizationContractArtifact } from '../src/config/quantization-contract-check.js';
import { buildRequiredInferenceFieldsContractArtifact } from '../src/config/required-inference-fields-contract-check.js';
import {
  getInferenceExecutionRulesContractArtifact,
  getInferenceLayerPatternContractArtifact,
} from '../src/rules/rule-registry.js';
import { buildSummary, loadConversionReports } from './summarize-conversion-reports.js';

function parseArgs(argv) {
  const args = {
    json: false,
    reportsRoot: '',
    failOnReportContracts: false,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--json') {
      args.json = true;
      continue;
    }
    if (arg === '--reports-root') {
      args.reportsRoot = String(argv[index + 1] ?? '').trim();
      index += 1;
      continue;
    }
    if (arg === '--fail-on-report-contracts') {
      args.failOnReportContracts = true;
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      args.help = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function usage() {
  return [
    'Usage:',
    '  node tools/check-contract-artifacts.js [--json] [--reports-root <dir>] [--fail-on-report-contracts]',
  ].join('\n');
}

function summarizeArtifact(id, artifact) {
  const checks = Array.isArray(artifact?.checks) ? artifact.checks : [];
  return {
    id,
    ok: artifact?.ok === true,
    checks: checks.length,
    passedChecks: checks.filter((entry) => entry?.ok === true).length,
    firstError: Array.isArray(artifact?.errors) && artifact.errors.length > 0
      ? String(artifact.errors[0])
      : null,
    stats: artifact?.stats ?? null,
  };
}

async function buildContractSummary(args) {
  const artifacts = [
    summarizeArtifact('kernelPath', getKernelPathContractArtifact()),
    summarizeArtifact('executionRules', getInferenceExecutionRulesContractArtifact()),
    summarizeArtifact('layerPattern', getInferenceLayerPatternContractArtifact()),
    summarizeArtifact('merge', buildMergeContractArtifact()),
    summarizeArtifact('quantization', buildQuantizationContractArtifact()),
    summarizeArtifact('requiredInferenceFields', buildRequiredInferenceFieldsContractArtifact()),
  ];
  let reports = null;
  if (args.reportsRoot) {
    const entries = await loadConversionReports(args.reportsRoot);
    reports = buildSummary(entries, args.reportsRoot, 10);
  }
  const ok = artifacts.every((entry) => entry.ok === true)
    && (
      !args.failOnReportContracts
      || reports == null
      || (
        reports.contractFail === 0
        && reports.graphFail === 0
        && reports.layerPatternFail === 0
        && reports.requiredInferenceFail === 0
      )
    );
  return {
    schemaVersion: 1,
    source: 'doppler',
    ok,
    artifacts,
    reports,
  };
}

function printHuman(summary) {
  console.log(`[contracts] status=${summary.ok ? 'pass' : 'fail'} artifacts=${summary.artifacts.length}`);
  for (const artifact of summary.artifacts) {
    console.log(
      `[contracts] ${artifact.id} status=${artifact.ok ? 'pass' : 'fail'} ` +
      `checks=${artifact.passedChecks}/${artifact.checks}`
    );
    if (!artifact.ok && artifact.firstError) {
      console.log(`[contracts] ${artifact.id}_error=${JSON.stringify(artifact.firstError)}`);
    }
  }
  if (summary.reports) {
    console.log(
      `[contracts] reports total=${summary.reports.totalReports} ` +
      `contractFail=${summary.reports.contractFail} graphFail=${summary.reports.graphFail} ` +
      `layerPatternFail=${summary.reports.layerPatternFail} requiredInferenceFail=${summary.reports.requiredInferenceFail}`
    );
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const summary = await buildContractSummary(args);
  if (args.json) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    printHuman(summary);
  }
  if (!summary.ok) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
