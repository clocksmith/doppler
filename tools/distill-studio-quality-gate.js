#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

function normalizeString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function parseNumber(value, fallback = null) {
  if (value === undefined || value === null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Expected finite number, received: ${value}`);
  }
  return parsed;
}

function parseArgs(argv) {
  const parsed = {
    report: null,
    outDir: null,
    minSteps: 1,
    maxTotalLoss: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = String(argv[i] || '');
    const nextValue = () => {
      const value = argv[i + 1];
      if (value == null || String(value).startsWith('--')) {
        throw new Error(`Missing value for ${token}`);
      }
      i += 1;
      return value;
    };
    if (token === '--report') {
      parsed.report = String(nextValue()).trim();
      continue;
    }
    if (token === '--out-dir') {
      parsed.outDir = String(nextValue()).trim();
      continue;
    }
    if (token === '--min-steps') {
      parsed.minSteps = parseNumber(nextValue());
      continue;
    }
    if (token === '--max-total-loss') {
      parsed.maxTotalLoss = parseNumber(nextValue(), null);
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }

  const report = normalizeString(parsed.report);
  if (!report) {
    throw new Error('--report is required');
  }
  const resolvedReport = resolve(report);
  const resolvedOutDir = resolve(parsed.outDir || dirname(resolvedReport));
  return {
    reportPath: resolvedReport,
    outDir: resolvedOutDir,
    minSteps: Math.max(1, Math.floor(parsed.minSteps ?? 1)),
    maxTotalLoss: parsed.maxTotalLoss,
  };
}

function readByPath(obj, path) {
  let current = obj;
  for (const key of path) {
    if (!current || typeof current !== 'object') return null;
    current = current[key];
  }
  return current;
}

function resolveReportModel(payload) {
  const candidates = [
    readByPath(payload, ['result', 'report']),
    payload.report,
    readByPath(payload, ['result']),
    payload,
  ];
  for (const candidate of candidates) {
    if (candidate && typeof candidate === 'object') {
      return candidate;
    }
  }
  throw new Error('Unable to resolve report payload object');
}

function computeReportId(reportObj, rawJson) {
  const explicit = normalizeString(reportObj?.reportId || reportObj?.id || reportObj?.report?.id);
  if (explicit) return explicit;
  return createHash('sha256').update(rawJson).digest('hex');
}

function buildLanguageGate({
  language,
  direction,
  reportId,
  modelId,
  trainingSchemaVersion,
  runtimeProfile,
  kernelPath,
  seed,
  steps,
  finalTotalLoss,
  directionCount,
  minSteps,
  maxTotalLoss,
}) {
  const checks = [
    {
      key: 'steps_minimum',
      pass: Number.isFinite(steps) && steps >= minSteps,
      value: steps,
      target: `>=${minSteps}`,
    },
    {
      key: 'direction_observed',
      pass: Number.isFinite(directionCount) && directionCount > 0,
      value: directionCount,
      target: '>0',
    },
    {
      key: 'loss_finite',
      pass: Number.isFinite(finalTotalLoss),
      value: finalTotalLoss,
      target: 'finite',
    },
  ];
  if (Number.isFinite(maxTotalLoss)) {
    checks.push({
      key: 'loss_threshold',
      pass: Number.isFinite(finalTotalLoss) && finalTotalLoss <= maxTotalLoss,
      value: finalTotalLoss,
      target: `<=${maxTotalLoss}`,
    });
  }

  return {
    schemaVersion: 1,
    artifactType: 'distill_quality_gate',
    language,
    direction,
    reportId,
    modelId,
    trainingSchemaVersion,
    runtimeProfile,
    kernelPath,
    seed,
    metrics: {
      steps,
      finalTotalLoss,
      directionCount,
    },
    checks,
    pass: checks.every((entry) => entry.pass === true),
    generatedAt: new Date().toISOString(),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const raw = await readFile(args.reportPath, 'utf8');
  const parsed = JSON.parse(raw);
  const reportObj = resolveReportModel(parsed);
  const reportId = computeReportId(reportObj, raw);

  const metrics = (
    (reportObj.metrics && typeof reportObj.metrics === 'object')
      ? reportObj.metrics
      : {}
  );
  const trainingReport = Array.isArray(metrics.trainingMetricsReport)
    ? metrics.trainingMetricsReport
    : [];
  const finalEntry = trainingReport.length > 0 ? trainingReport[trainingReport.length - 1] : {};
  const directionCounts = (
    metrics.distillDataset && typeof metrics.distillDataset === 'object' && metrics.distillDataset.directionCounts
  ) || {};
  const runtimeProfile = normalizeString(reportObj.runtimeProfile);
  const kernelPath = normalizeString(finalEntry.kernel_path);
  const seed = Number.isFinite(finalEntry.seed) ? finalEntry.seed : null;
  const finalTotalLoss = Number.isFinite(finalEntry.total_loss) ? finalEntry.total_loss : null;
  const modelId = normalizeString(reportObj.modelId) || 'training';
  const trainingSchemaVersion = Number.isFinite(metrics.trainingSchemaVersion)
    ? metrics.trainingSchemaVersion
    : null;

  const enGate = buildLanguageGate({
    language: 'en',
    direction: 'en->es',
    reportId,
    modelId,
    trainingSchemaVersion,
    runtimeProfile,
    kernelPath,
    seed,
    steps: trainingReport.length,
    finalTotalLoss,
    directionCount: Number(directionCounts['en->es'] || 0),
    minSteps: args.minSteps,
    maxTotalLoss: args.maxTotalLoss,
  });
  const esGate = buildLanguageGate({
    language: 'es',
    direction: 'es->en',
    reportId,
    modelId,
    trainingSchemaVersion,
    runtimeProfile,
    kernelPath,
    seed,
    steps: trainingReport.length,
    finalTotalLoss,
    directionCount: Number(directionCounts['es->en'] || 0),
    minSteps: args.minSteps,
    maxTotalLoss: args.maxTotalLoss,
  });
  const reproducibilityBundle = {
    schemaVersion: 1,
    artifactType: 'distill_reproducibility_bundle',
    reportId,
    reportPath: args.reportPath,
    modelId,
    trainingSchemaVersion,
    runtimeProfile,
    kernelPath,
    seed,
    generatedAt: new Date().toISOString(),
    checkpointResumeTimelineLength: Array.isArray(metrics.checkpointResumeTimeline)
      ? metrics.checkpointResumeTimeline.length
      : 0,
    gateFiles: [
      'distill-quality-gate-en.json',
      'distill-quality-gate-es.json',
    ],
  };

  await mkdir(args.outDir, { recursive: true });
  const enPath = join(args.outDir, 'distill-quality-gate-en.json');
  const esPath = join(args.outDir, 'distill-quality-gate-es.json');
  const bundlePath = join(args.outDir, 'distill-reproducibility-bundle.json');
  await writeFile(enPath, `${JSON.stringify(enGate, null, 2)}\n`, 'utf8');
  await writeFile(esPath, `${JSON.stringify(esGate, null, 2)}\n`, 'utf8');
  await writeFile(bundlePath, `${JSON.stringify(reproducibilityBundle, null, 2)}\n`, 'utf8');

  console.log(`[distill-quality-gate] wrote ${enPath}`);
  console.log(`[distill-quality-gate] wrote ${esPath}`);
  console.log(`[distill-quality-gate] wrote ${bundlePath}`);
}

await main();
