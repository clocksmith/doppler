#!/usr/bin/env node

import { pathToFileURL } from 'node:url';

import { buildMergeContractArtifact } from '../src/config/merge-contract-check.js';
import { getKernelPathContractArtifact } from '../src/config/kernel-path-loader.js';
import { buildQuantizationContractArtifact } from '../src/config/quantization-contract-check.js';
import { buildRequiredInferenceFieldsContractArtifact } from '../src/config/required-inference-fields-contract-check.js';
import {
  getInferenceExecutionRulesContractArtifact,
  getInferenceLayerPatternContractArtifact,
} from '../src/rules/rule-registry.js';
import { buildSummary, loadConversionReports } from './summarize-conversion-reports.js';
import { buildKernelPathBuilderContractArtifact } from './sync-kernel-path-builder-index.js';
import { runSweep as runLeanExecutionContractManifestSweep } from './lean-execution-contract-sweep.js';
import { runSweep as runLeanExecutionContractConfigSweep } from './lean-execution-contract-config-sweep.js';

function parseArgs(argv) {
  const args = {
    json: false,
    reportsRoot: '',
    failOnReportContracts: false,
    withLean: false,
    leanCheck: true,
    leanManifestRoot: 'models',
    leanConfigRoot: 'src/config/conversion',
    leanFixtureMap: 'tests/fixtures/lean-execution-contract-fixtures.json',
    leanRequireManifestMatch: false,
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
    if (arg === '--with-lean') {
      args.withLean = true;
      continue;
    }
    if (arg === '--lean-no-check') {
      args.leanCheck = false;
      continue;
    }
    if (arg === '--lean-manifest-root') {
      args.leanManifestRoot = String(argv[index + 1] ?? '').trim() || args.leanManifestRoot;
      index += 1;
      continue;
    }
    if (arg === '--lean-config-root') {
      args.leanConfigRoot = String(argv[index + 1] ?? '').trim() || args.leanConfigRoot;
      index += 1;
      continue;
    }
    if (arg === '--lean-fixture-map') {
      args.leanFixtureMap = String(argv[index + 1] ?? '').trim() || args.leanFixtureMap;
      index += 1;
      continue;
    }
    if (arg === '--lean-require-manifest-match') {
      args.leanRequireManifestMatch = true;
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
    '  node tools/check-contract-artifacts.js [--json] [--reports-root <dir>] [--fail-on-report-contracts] [--with-lean] [--lean-no-check] [--lean-manifest-root <dir>] [--lean-config-root <dir>] [--lean-fixture-map <json>] [--lean-require-manifest-match]',
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

function summarizeLeanSweep(id, summary) {
  const results = Array.isArray(summary?.results) ? summary.results : [];
  const firstFailure = results.find((entry) => entry?.status === 'fail' || entry?.status === 'error') || null;
  return {
    id,
    ok: summary?.ok === true,
    checks: results.length,
    passedChecks: results.filter((entry) => entry?.status === 'pass').length,
    firstError: firstFailure?.reason ?? null,
    stats: summary?.totals ?? null,
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
    summarizeArtifact('kernelPathBuilder', await buildKernelPathBuilderContractArtifact()),
  ];
  let lean = null;
  if (args.withLean) {
    const manifestSweep = await runLeanExecutionContractManifestSweep(args.leanManifestRoot, {
      check: args.leanCheck,
    });
    const configSweep = await runLeanExecutionContractConfigSweep({
      configRoot: args.leanConfigRoot,
      manifestRoot: args.leanManifestRoot,
      fixtureMap: args.leanFixtureMap,
      check: args.leanCheck,
      requireManifestMatch: args.leanRequireManifestMatch,
    });
    lean = {
      manifestSweep,
      configSweep,
    };
    artifacts.push(summarizeLeanSweep('leanExecutionContractManifests', manifestSweep));
    artifacts.push(summarizeLeanSweep('leanExecutionContractConfigs', configSweep));
  }
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
    lean,
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

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}

export {
  buildContractSummary,
};
