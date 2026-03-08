#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

function parsePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseOptionalPositiveInteger(value, label) {
  if (value === undefined || value === null || value === '') return null;
  return parsePositiveInteger(value, label);
}

function parseOptionalString(value) {
  if (value === undefined || value === null) return null;
  const normalized = String(value).trim();
  return normalized || null;
}

function parseStringList(value) {
  if (Array.isArray(value)) {
    const normalized = value.map((entry) => parseOptionalString(entry)).filter(Boolean);
    return normalized.length > 0 ? normalized : null;
  }
  const normalized = parseOptionalString(value);
  if (!normalized) return null;
  const list = normalized.split(',').map((entry) => entry.trim()).filter(Boolean);
  return list.length > 0 ? list : null;
}

function parseArgs(argv) {
  const parsed = {
    help: false,
    mode: 'bench',
    surface: 'node',
    outDir: 'reports/training/distill',
    workload: 'tiny',
    trainingBenchSteps: null,
    stageASteps: null,
    stageBSteps: null,
    checkpointEvery: null,
    distillDatasetPath: null,
    distillSourceLangs: null,
    distillTargetLangs: null,
    distillPairAllowlist: null,
    strictPairContract: false,
    distillShardIndex: null,
    distillShardCount: null,
    resumeFrom: null,
    forceResume: false,
    forceResumeReason: null,
    forceResumeSource: null,
    checkpointOperator: null,
    skipStageA: false,
    stageAOnly: false,
    stageAArtifact: null,
    stageAArtifactHash: null,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      parsed.help = true;
      continue;
    }
    if (arg === '--mode') {
      parsed.mode = String(argv[i + 1] || parsed.mode);
      i += 1;
      continue;
    }
    if (arg === '--surface') {
      parsed.surface = String(argv[i + 1] || parsed.surface);
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
    if (arg === '--training-bench-steps') {
      parsed.trainingBenchSteps = parsePositiveInteger(argv[i + 1], '--training-bench-steps');
      i += 1;
      continue;
    }
    if (arg === '--stage-a-steps') {
      parsed.stageASteps = parsePositiveInteger(argv[i + 1], '--stage-a-steps');
      i += 1;
      continue;
    }
    if (arg === '--stage-b-steps') {
      parsed.stageBSteps = parsePositiveInteger(argv[i + 1], '--stage-b-steps');
      i += 1;
      continue;
    }
    if (arg === '--checkpoint-every') {
      parsed.checkpointEvery = parsePositiveInteger(argv[i + 1], '--checkpoint-every');
      i += 1;
      continue;
    }
    if (arg === '--distill-dataset-path') {
      parsed.distillDatasetPath = parseOptionalString(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--distill-source-langs') {
      parsed.distillSourceLangs = parseStringList(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--distill-target-langs') {
      parsed.distillTargetLangs = parseStringList(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--distill-pair-allowlist') {
      parsed.distillPairAllowlist = parseStringList(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--strict-pair-contract') {
      parsed.strictPairContract = true;
      continue;
    }
    if (arg === '--distill-shard-index') {
      parsed.distillShardIndex = parsePositiveInteger(argv[i + 1], '--distill-shard-index');
      i += 1;
      continue;
    }
    if (arg === '--distill-shard-count') {
      parsed.distillShardCount = parsePositiveInteger(argv[i + 1], '--distill-shard-count');
      i += 1;
      continue;
    }
    if (arg === '--resume-from') {
      parsed.resumeFrom = parseOptionalString(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--force-resume') {
      parsed.forceResume = true;
      continue;
    }
    if (arg === '--force-resume-reason') {
      parsed.forceResumeReason = parseOptionalString(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--force-resume-source') {
      parsed.forceResumeSource = parseOptionalString(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--checkpoint-operator') {
      parsed.checkpointOperator = parseOptionalString(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--skip-stage-a') {
      parsed.skipStageA = true;
      continue;
    }
    if (arg === '--stage-a-only') {
      parsed.stageAOnly = true;
      continue;
    }
    if (arg === '--stage-a-artifact') {
      parsed.stageAArtifact = parseOptionalString(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--stage-a-artifact-hash') {
      parsed.stageAArtifactHash = parseOptionalString(argv[i + 1]);
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }

  if (parsed.mode !== 'bench' && parsed.mode !== 'train') {
    throw new Error('--mode must be "bench" or "train"');
  }
  if (parsed.forceResumeReason && !parsed.forceResume) {
    throw new Error('--force-resume-reason requires --force-resume');
  }
  if (parsed.forceResumeSource && !parsed.forceResume) {
    throw new Error('--force-resume-source requires --force-resume');
  }
  if (parsed.checkpointOperator && !parsed.forceResume) {
    throw new Error('--checkpoint-operator requires --force-resume');
  }
  if (parsed.stageAOnly && parsed.skipStageA) {
    throw new Error('--stage-a-only cannot be combined with --skip-stage-a');
  }

  return parsed;
}

function usage() {
  return [
    'Usage:',
    '  node tools/run-distill-bench.mjs [flags]',
    '',
    'Modes:',
    '  --mode bench|train                Default: bench',
    '',
    'Common flags:',
    '  --surface node|browser            Default: node',
    '  --workload tiny|medium|<path>     Default: tiny',
    '  --out-dir <dir>                   Default: reports/training/distill',
    '  --distill-dataset-path <path>',
    '  --distill-source-langs <csv>',
    '  --distill-target-langs <csv>',
    '  --distill-pair-allowlist <csv>',
    '  --strict-pair-contract',
    '  --distill-shard-index <int>',
    '  --distill-shard-count <int>',
    '  --resume-from <checkpoint-path>',
    '  --checkpoint-every <int>',
    '',
    'Resume override flags:',
    '  --force-resume',
    '  --force-resume-reason <text>',
    '  --force-resume-source <text>',
    '  --checkpoint-operator <text>',
    '',
    'Bench-mode flags:',
    '  --training-bench-steps <int>      Applies to stage_a + stage_b bench runs',
    '',
    'Train-mode flags:',
    '  --stage-a-steps <int>             Default: trainingBenchSteps',
    '  --stage-b-steps <int>             Default: trainingBenchSteps',
    '  --stage-a-only',
    '  --skip-stage-a                    Requires --stage-a-artifact',
    '  --stage-a-artifact <path>',
    '  --stage-a-artifact-hash <sha256>',
  ].join('\n');
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
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
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

function resolveOptionalInteger(value, fallback, label) {
  if (value === undefined || value === null || value === '') return fallback;
  return parsePositiveInteger(value, label);
}

function resolveResumeOverrideFields(args, workloadConfig) {
  const forceResume = args.forceResume || workloadConfig.forceResume === true;
  const forceResumeReason = args.forceResumeReason
    || parseOptionalString(workloadConfig.forceResumeReason)
    || null;
  const forceResumeSource = args.forceResumeSource
    || parseOptionalString(workloadConfig.forceResumeSource)
    || null;
  const checkpointOperator = args.checkpointOperator
    || parseOptionalString(workloadConfig.checkpointOperator)
    || null;

  if (!forceResume) {
    if (forceResumeReason || forceResumeSource || checkpointOperator) {
      throw new Error('force resume metadata requires forceResume=true.');
    }
    return {};
  }

  return {
    forceResume: true,
    forceResumeReason: forceResumeReason || 'operator_requested_resume_override',
    forceResumeSource: forceResumeSource || null,
    checkpointOperator,
  };
}

function createBaseTrainingRequest(params) {
  const request = {
    trainingSchemaVersion: params.trainingSchemaVersion,
    teacherModelId: params.teacherModelId,
    studentModelId: params.studentModelId,
    distillDatasetId: params.distillDatasetId,
    distillDatasetPath: params.distillDatasetPath,
    distillLanguagePair: params.distillLanguagePair,
    distillSourceLangs: params.distillSourceLangs,
    distillTargetLangs: params.distillTargetLangs,
    distillPairAllowlist: params.distillPairAllowlist,
    strictPairContract: params.strictPairContract,
    distillShardIndex: params.distillShardIndex,
    distillShardCount: params.distillShardCount,
    resumeFrom: params.resumeFrom,
    distillArtifactDir: params.outDirAbs,
    checkpointEvery: params.checkpointEvery,
    ...params.resumeOverrides,
  };
  return request;
}

async function runVerifyStage(stage, options) {
  return runCli([
    'verify',
    '--config',
    JSON.stringify({
      request: {
        suite: 'training',
        trainingStage: stage,
        trainingTests: [stage === 'stage_b' ? 'distill-stage-b' : 'distill-stage-a'],
        trainingBenchSteps: options.trainingBenchSteps,
        ...options.request,
        ...(stage === 'stage_b'
          ? {
            stageAArtifact: options.stageAArtifact,
            stageAArtifactHash: options.stageAArtifactHash,
          }
          : {}),
      },
      run: {
        surface: options.surface,
      },
    }),
    '--json',
  ]);
}

async function runBenchStage(stage, options) {
  return runCli([
    'bench',
    '--config',
    JSON.stringify({
      request: {
        workloadType: 'training',
        trainingStage: stage,
        trainingBenchSteps: options.trainingBenchSteps,
        ...options.request,
        ...(stage === 'stage_b'
          ? {
            stageAArtifact: options.stageAArtifact,
            stageAArtifactHash: options.stageAArtifactHash,
          }
          : {}),
      },
      run: {
        surface: options.surface,
      },
    }),
    '--runtime-config',
    options.runtimeConfigJson,
    '--json',
  ]);
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const outDirAbs = resolve(args.outDir);
  const workload = await loadWorkloadConfig(args.workload);

  if (!workload.config.trainingSchemaVersion) {
    throw new Error(`Workload config at ${workload.path} is missing required field: trainingSchemaVersion`);
  }
  if (!workload.config.trainingBenchSteps) {
    throw new Error(`Workload config at ${workload.path} is missing required field: trainingBenchSteps`);
  }
  if (!workload.config.teacherModelId) {
    throw new Error(`Workload config at ${workload.path} is missing required field: teacherModelId`);
  }
  if (!workload.config.studentModelId) {
    throw new Error(`Workload config at ${workload.path} is missing required field: studentModelId`);
  }
  if (!workload.config.distillDatasetId) {
    throw new Error(`Workload config at ${workload.path} is missing required field: distillDatasetId`);
  }
  const trainingSchemaVersion = Number(workload.config.trainingSchemaVersion);
  const defaultBenchSteps = resolveOptionalInteger(
    workload.config.trainingBenchSteps,
    null,
    'trainingBenchSteps'
  );
  const benchSteps = args.trainingBenchSteps ?? defaultBenchSteps;
  const stageASteps = args.stageASteps ?? resolveOptionalInteger(
    workload.config.stageASteps,
    benchSteps,
    'stageASteps'
  );
  const stageBSteps = args.stageBSteps ?? resolveOptionalInteger(
    workload.config.stageBSteps,
    benchSteps,
    'stageBSteps'
  );
  const checkpointEvery = args.checkpointEvery ?? parseOptionalPositiveInteger(
    workload.config.checkpointEvery,
    'checkpointEvery'
  );
  const teacherModelId = String(workload.config.teacherModelId);
  const studentModelId = String(workload.config.studentModelId);
  const distillDatasetId = String(workload.config.distillDatasetId);
  const distillDatasetPath = args.distillDatasetPath || (
    workload.config.distillDatasetPath
      ? String(workload.config.distillDatasetPath)
      : null
  );
  const distillSourceLangs = args.distillSourceLangs || parseStringList(workload.config.distillSourceLangs);
  const distillTargetLangs = args.distillTargetLangs || parseStringList(workload.config.distillTargetLangs);
  const distillPairAllowlist = args.distillPairAllowlist || parseStringList(workload.config.distillPairAllowlist);
  const strictPairContract = args.strictPairContract || workload.config.strictPairContract === true;
  const distillShardIndex = args.distillShardIndex
    ?? parseOptionalPositiveInteger(workload.config.distillShardIndex, 'distillShardIndex');
  const distillShardCount = args.distillShardCount
    ?? parseOptionalPositiveInteger(workload.config.distillShardCount, 'distillShardCount');
  const resumeFrom = args.resumeFrom || parseOptionalString(workload.config.resumeFrom);
  const stageAArtifactInput = args.stageAArtifact || parseOptionalString(workload.config.stageAArtifact);
  const stageAArtifactHashInput = args.stageAArtifactHash || parseOptionalString(workload.config.stageAArtifactHash);
  if (
    Number.isInteger(distillShardIndex)
    && Number.isInteger(distillShardCount)
    && distillShardIndex > distillShardCount
  ) {
    throw new Error('distillShardIndex must be <= distillShardCount');
  }
  if (!workload.config.distillLanguagePair) {
    throw new Error(`Workload config at ${workload.path} is missing required field: distillLanguagePair`);
  }
  const distillLanguagePair = String(workload.config.distillLanguagePair);
  const runtimeConfigJson = JSON.stringify({
    shared: {
      benchmark: {
        run: workload.config.benchRun || {},
      },
    },
  });
  const resumeOverrides = resolveResumeOverrideFields(args, workload.config);
  const baseRequest = createBaseTrainingRequest({
    trainingSchemaVersion,
    teacherModelId,
    studentModelId,
    distillDatasetId,
    distillDatasetPath,
    distillLanguagePair,
    distillSourceLangs,
    distillTargetLangs,
    distillPairAllowlist,
    strictPairContract,
    distillShardIndex,
    distillShardCount,
    resumeFrom,
    checkpointEvery,
    outDirAbs,
    resumeOverrides,
  });

  await mkdir(outDirAbs, { recursive: true });

  if (args.mode === 'bench') {
    const stageAVerify = await runVerifyStage('stage_a', {
      surface: args.surface,
      trainingBenchSteps: null,
      request: baseRequest,
    });
    const stageAArtifact = extractStageAArtifact(stageAVerify);
    const stageABench = await runBenchStage('stage_a', {
      surface: args.surface,
      trainingBenchSteps: benchSteps,
      request: baseRequest,
      runtimeConfigJson,
    });
    const stageBBench = await runBenchStage('stage_b', {
      surface: args.surface,
      trainingBenchSteps: benchSteps,
      request: baseRequest,
      runtimeConfigJson,
      stageAArtifact: stageAArtifact.stageAArtifact,
      stageAArtifactHash: String(stageAArtifact.stageAArtifactHash || ''),
    });

    const summary = {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      mode: 'bench',
      claimBoundary: 'TranslateGemma distill pipeline for operational benchmarking; not claim of paper-parity SOTA.',
      workload: {
        path: workload.path,
        config: workload.config,
      },
      resolved: {
        benchSteps,
        checkpointEvery,
        distillDatasetPath,
        distillSourceLangs,
        distillTargetLangs,
        distillPairAllowlist,
        strictPairContract,
        distillShardIndex,
        distillShardCount,
        resumeFrom,
        ...resumeOverrides,
      },
      stageAArtifact,
      stageAVerify,
      stageABench,
      stageBBench,
    };
    const outPath = join(outDirAbs, 'distill-bench-summary.json');
    await writeFile(outPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
    process.stdout.write(`${JSON.stringify({ ok: true, mode: 'bench', outPath }, null, 2)}\n`);
    return;
  }

  if (args.skipStageA && !stageAArtifactInput) {
    throw new Error('--skip-stage-a requires --stage-a-artifact (or workload stageAArtifact).');
  }

  let stageAResult = null;
  let stageAArtifact = stageAArtifactInput
    ? {
      stageAArtifact: stageAArtifactInput,
      stageAArtifactHash: stageAArtifactHashInput || null,
    }
    : null;
  if (!args.skipStageA) {
    stageAResult = await runVerifyStage('stage_a', {
      surface: args.surface,
      trainingBenchSteps: stageASteps,
      request: baseRequest,
    });
    stageAArtifact = extractStageAArtifact(stageAResult);
  }

  let stageBResult = null;
  if (!args.stageAOnly) {
    if (!stageAArtifact?.stageAArtifact) {
      throw new Error('stage_b requires stage_a artifact.');
    }
    stageBResult = await runVerifyStage('stage_b', {
      surface: args.surface,
      trainingBenchSteps: stageBSteps,
      request: baseRequest,
      stageAArtifact: stageAArtifact.stageAArtifact,
      stageAArtifactHash: String(stageAArtifact.stageAArtifactHash || ''),
    });
  }

  const summary = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    mode: 'train',
    claimBoundary: 'TranslateGemma distill pipeline for operational long-run validation and reproducible lineage.',
    workload: {
      path: workload.path,
      config: workload.config,
    },
    resolved: {
      stageASteps: args.skipStageA ? 0 : stageASteps,
      stageBSteps: args.stageAOnly ? 0 : stageBSteps,
      checkpointEvery,
      skipStageA: args.skipStageA,
      stageAOnly: args.stageAOnly,
      distillDatasetPath,
      distillSourceLangs,
      distillTargetLangs,
      distillPairAllowlist,
      strictPairContract,
      distillShardIndex,
      distillShardCount,
      resumeFrom,
      ...resumeOverrides,
    },
    stageAArtifact,
    stageAResult,
    stageBResult,
  };
  const outPath = join(outDirAbs, 'distill-train-summary.json');
  await writeFile(outPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify({ ok: true, mode: 'train', outPath }, null, 2)}\n`);
}

await main();
