#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

function requireFlagValue(argv, index, flag) {
  const value = argv[index + 1];
  if (value == null || String(value).startsWith('--')) {
    throw new Error(`Missing value for ${flag}`);
  }
  return String(value);
}

export function parseArgs(argv) {
  const parsed = {
    surface: 'node',
    outDir: 'reports/training/ul',
    workload: 'tiny',
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--surface') {
      parsed.surface = requireFlagValue(argv, i, '--surface');
      i += 1;
      continue;
    }
    if (arg === '--out-dir') {
      parsed.outDir = requireFlagValue(argv, i, '--out-dir');
      i += 1;
      continue;
    }
    if (arg === '--workload') {
      parsed.workload = requireFlagValue(argv, i, '--workload');
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }
  return parsed;
}

function buildRuntimeConfigJson(benchRun) {
  if (benchRun == null) {
    return null;
  }
  if (!benchRun || typeof benchRun !== 'object' || Array.isArray(benchRun)) {
    throw new Error('Workload field "benchRun" must be an object when provided.');
  }
  return JSON.stringify({
    shared: {
      benchmark: {
        run: benchRun,
      },
    },
  });
}

export async function loadWorkloadConfig(workloadArg) {
  const presets = {
    tiny: 'tools/configs/training-workloads/ul-training-tiny.json',
    medium: 'tools/configs/training-workloads/ul-training-medium.json',
  };
  const candidate = presets[workloadArg] || workloadArg;
  const absolute = resolve(candidate);
  const raw = await readFile(absolute, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object') {
    throw new Error(`Invalid UL workload config at ${absolute}`);
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

function extractStage1Artifact(trainingResult) {
  const result = trainingResult?.result;
  const entries = Array.isArray(result?.results) ? result.results : [];
  for (const entry of entries) {
    const artifact = entry?.artifact;
    if (artifact?.manifestPath) {
      return {
        stage1Artifact: artifact.manifestPath,
        stage1ArtifactHash: artifact.manifestFileHash || artifact.manifestHash || null,
      };
    }
  }
  throw new Error('Unable to find stage1 artifact in training suite output.');
}

async function main() {
  const args = parseArgs(process.argv);
  const outDirAbs = resolve(args.outDir);
  const workload = await loadWorkloadConfig(args.workload);
  if (!workload.config.trainingSchemaVersion) {
    throw new Error(`Workload config at ${workload.path} is missing required field: trainingSchemaVersion`);
  }
  if (!workload.config.trainingBenchSteps) {
    throw new Error(`Workload config at ${workload.path} is missing required field: trainingBenchSteps`);
  }
  if (workload.config.seed == null) {
    throw new Error(`Workload config at ${workload.path} is missing required field: seed`);
  }
  if (!Array.isArray(workload.config.trainingTests) || workload.config.trainingTests.length === 0) {
    throw new Error(`Workload config at ${workload.path} is missing required field: trainingTests (non-empty array)`);
  }
  const trainingSchemaVersion = Number(workload.config.trainingSchemaVersion);
  const trainingBenchSteps = Number(workload.config.trainingBenchSteps);
  const seed = Number(workload.config.seed);
  const trainingTests = workload.config.trainingTests.map((value) => String(value));
  if (!trainingTests.includes('ul-stage1') || !trainingTests.includes('ul-stage2')) {
    throw new Error(`Workload config at ${workload.path} must include trainingTests: ["ul-stage1", "ul-stage2"]`);
  }
  const stage1TestId = 'ul-stage1';
  const runtimeConfigJson = buildRuntimeConfigJson(workload.config.benchRun);
  const trainingConfig = {
    ul: {
      seed,
    },
  };
  await mkdir(outDirAbs, { recursive: true });

  const stage1Verify = await runCli([
    'verify',
    '--config',
    JSON.stringify({
      request: {
        suite: 'training',
        trainingSchemaVersion,
        trainingConfig,
        trainingStage: 'stage1_joint',
        trainingTests: [stage1TestId],
      },
      run: {
        surface: args.surface,
      },
    }),
    '--json',
  ]);
  const stage1Artifact = extractStage1Artifact(stage1Verify);

  const stage1BenchArgs = [
    'bench',
    '--config',
    JSON.stringify({
      request: {
        trainingSchemaVersion,
        trainingBenchSteps,
        trainingConfig,
        workloadType: 'training',
        trainingStage: 'stage1_joint',
      },
      run: {
        surface: args.surface,
      },
    }),
    '--json',
  ];
  if (runtimeConfigJson !== null) {
    stage1BenchArgs.splice(stage1BenchArgs.length - 1, 0, '--runtime-config', runtimeConfigJson);
  }
  const stage1Bench = await runCli(stage1BenchArgs);

  const stage2BenchArgs = [
    'bench',
    '--config',
    JSON.stringify({
      request: {
        trainingSchemaVersion,
        trainingBenchSteps,
        trainingConfig,
        workloadType: 'training',
        trainingStage: 'stage2_base',
        stage1Artifact: stage1Artifact.stage1Artifact,
        stage1ArtifactHash: String(stage1Artifact.stage1ArtifactHash || ''),
      },
      run: {
        surface: args.surface,
      },
    }),
    '--json',
  ];
  if (runtimeConfigJson !== null) {
    stage2BenchArgs.splice(stage2BenchArgs.length - 1, 0, '--runtime-config', runtimeConfigJson);
  }
  const stage2Bench = await runCli(stage2BenchArgs);

  const summary = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    claimBoundary: 'UL-inspired practical two-stage pipeline; not paper-equivalent SOTA.',
    workload: {
      path: workload.path,
      config: workload.config,
    },
    stage1Artifact,
    stage1Verify,
    stage1Bench,
    stage2Bench,
  };
  const outPath = join(outDirAbs, 'ul-bench-summary.json');
  await writeFile(outPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');

  process.stdout.write(`${JSON.stringify({ ok: true, outPath }, null, 2)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
