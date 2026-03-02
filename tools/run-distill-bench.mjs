#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';

function parseArgs(argv) {
  const parsed = {
    surface: 'node',
    outDir: 'bench/out/distill',
    workload: 'tiny',
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--surface') {
      parsed.surface = String(argv[i + 1] || 'node');
      i += 1;
      continue;
    }
    if (arg === '--out-dir') {
      parsed.outDir = String(argv[i + 1] || parsed.outDir);
      i += 1;
      continue;
    }
    if (arg === '--workload') {
      parsed.workload = String(argv[i + 1] || parsed.workload);
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }
  return parsed;
}

async function loadWorkloadConfig(workloadArg) {
  const presets = {
    tiny: 'tools/configs/training-workloads/distill-translategemma-tiny.json',
    medium: 'tools/configs/training-workloads/distill-translategemma-medium.json',
  };
  const candidate = presets[workloadArg] || workloadArg;
  const absolute = resolve(candidate);
  const raw = await readFile(absolute, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object') {
    throw new Error(`Invalid distill workload config at ${absolute}`);
  }
  return { path: absolute, config: parsed };
}

async function runCli(args) {
  return new Promise((resolvePromise, rejectPromise) => {
    const proc = spawn('node', ['tools/doppler-cli.js', ...args], {
      stdio: ['ignore', 'pipe', 'pipe'],
      cwd: process.cwd(),
      env: process.env,
    });
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    proc.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    proc.on('error', rejectPromise);
    proc.on('close', (code) => {
      if (code !== 0) {
        rejectPromise(new Error(`doppler-cli exited ${code}\n${stderr || stdout}`));
        return;
      }
      try {
        resolvePromise(JSON.parse(stdout));
      } catch {
        rejectPromise(new Error(`Failed to parse doppler-cli JSON output:\n${stdout}`));
      }
    });
  });
}

function extractStageAArtifact(trainingResult) {
  const result = trainingResult?.result;
  const entries = Array.isArray(result?.results) ? result.results : [];
  for (const entry of entries) {
    const artifact = entry?.artifact;
    if (artifact?.manifestPath) {
      return {
        stageAArtifact: artifact.manifestPath,
        stageAArtifactHash: artifact.manifestFileHash || artifact.manifestHash || null,
      };
    }
  }
  throw new Error('Unable to find stage_a artifact in training suite output.');
}

async function main() {
  const args = parseArgs(process.argv);
  const outDirAbs = resolve(args.outDir);
  const workload = await loadWorkloadConfig(args.workload);
  const trainingSchemaVersion = Number(workload.config.trainingSchemaVersion || 1);
  const trainingBenchSteps = Number(workload.config.trainingBenchSteps || 2);
  const teacherModelId = String(workload.config.teacherModelId || 'translategemma-4b-it-wq4k-ef16-hf16');
  const studentModelId = String(workload.config.studentModelId || 'gemma-3-1b-it-wq4k-ef16-hf16');
  const distillDatasetId = String(workload.config.distillDatasetId || 'en-es');
  const distillLanguagePair = String(workload.config.distillLanguagePair || 'en-es');
  const trainingTests = Array.isArray(workload.config.trainingTests)
    ? workload.config.trainingTests.map((value) => String(value))
    : [];
  const stageATestId = trainingTests.includes('distill-stage-a') ? 'distill-stage-a' : 'distill-stage-a';

  const runtimeConfigJson = JSON.stringify({
    shared: {
      benchmark: {
        run: workload.config.benchRun || {},
      },
    },
  });

  await mkdir(outDirAbs, { recursive: true });

  const stageAVerify = await runCli([
    'test-model',
    '--suite',
    'training',
    '--surface',
    args.surface,
    '--training-schema-version',
    String(trainingSchemaVersion),
    '--training-stage',
    'stage_a',
    '--training-tests',
    stageATestId,
    '--teacher-model-id',
    teacherModelId,
    '--student-model-id',
    studentModelId,
    '--distill-dataset-id',
    distillDatasetId,
    '--distill-language-pair',
    distillLanguagePair,
    '--distill-artifact-dir',
    outDirAbs,
    '--json',
  ]);
  const stageAArtifact = extractStageAArtifact(stageAVerify);

  const stageABench = await runCli([
    'bench',
    '--surface',
    args.surface,
    '--training-schema-version',
    String(trainingSchemaVersion),
    '--training-bench-steps',
    String(trainingBenchSteps),
    '--runtime-config-json',
    runtimeConfigJson,
    '--workload-type',
    'training',
    '--training-stage',
    'stage_a',
    '--teacher-model-id',
    teacherModelId,
    '--student-model-id',
    studentModelId,
    '--distill-dataset-id',
    distillDatasetId,
    '--distill-language-pair',
    distillLanguagePair,
    '--distill-artifact-dir',
    outDirAbs,
    '--json',
  ]);

  const stageBBench = await runCli([
    'bench',
    '--surface',
    args.surface,
    '--training-schema-version',
    String(trainingSchemaVersion),
    '--training-bench-steps',
    String(trainingBenchSteps),
    '--runtime-config-json',
    runtimeConfigJson,
    '--workload-type',
    'training',
    '--training-stage',
    'stage_b',
    '--teacher-model-id',
    teacherModelId,
    '--student-model-id',
    studentModelId,
    '--distill-dataset-id',
    distillDatasetId,
    '--distill-language-pair',
    distillLanguagePair,
    '--distill-artifact-dir',
    outDirAbs,
    '--stagea-artifact',
    stageAArtifact.stageAArtifact,
    '--stagea-artifact-hash',
    String(stageAArtifact.stageAArtifactHash || ''),
    '--json',
  ]);

  const summary = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    claimBoundary: 'TranslateGemma distill pipeline for operational benchmarking; not claim of paper-parity SOTA.',
    workload: {
      path: workload.path,
      config: workload.config,
    },
    stageAArtifact,
    stageAVerify,
    stageABench,
    stageBBench,
  };
  const outPath = join(outDirAbs, 'distill-bench-summary.json');
  await writeFile(outPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');

  process.stdout.write(`${JSON.stringify({ ok: true, outPath }, null, 2)}\n`);
}

await main();
