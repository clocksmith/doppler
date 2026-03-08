#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadTrainingWorkloadPack } from '../src/training/workloads.js';

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

function requireFlagValue(argv, index, flag) {
  const value = argv[index + 1];
  if (value == null || String(value).startsWith('--')) {
    throw new Error(`Missing value for ${flag}`);
  }
  return value;
}

export function parseArgs(argv) {
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
      parsed.mode = String(requireFlagValue(argv, i, '--mode'));
      i += 1;
      continue;
    }
    if (arg === '--surface') {
      parsed.surface = String(requireFlagValue(argv, i, '--surface'));
      i += 1;
      continue;
    }
    if (arg === '--out-dir') {
      parsed.outDir = String(requireFlagValue(argv, i, '--out-dir'));
      i += 1;
      continue;
    }
    if (arg === '--workload') {
      parsed.workload = String(requireFlagValue(argv, i, '--workload'));
      i += 1;
      continue;
    }
    if (arg === '--training-bench-steps') {
      parsed.trainingBenchSteps = parsePositiveInteger(requireFlagValue(argv, i, '--training-bench-steps'), '--training-bench-steps');
      i += 1;
      continue;
    }
    if (arg === '--stage-a-steps') {
      parsed.stageASteps = parsePositiveInteger(requireFlagValue(argv, i, '--stage-a-steps'), '--stage-a-steps');
      i += 1;
      continue;
    }
    if (arg === '--stage-b-steps') {
      parsed.stageBSteps = parsePositiveInteger(requireFlagValue(argv, i, '--stage-b-steps'), '--stage-b-steps');
      i += 1;
      continue;
    }
    if (arg === '--checkpoint-every') {
      parsed.checkpointEvery = parsePositiveInteger(requireFlagValue(argv, i, '--checkpoint-every'), '--checkpoint-every');
      i += 1;
      continue;
    }
    if (arg === '--distill-dataset-path') {
      parsed.distillDatasetPath = parseOptionalString(requireFlagValue(argv, i, '--distill-dataset-path'));
      i += 1;
      continue;
    }
    if (arg === '--distill-source-langs') {
      parsed.distillSourceLangs = parseStringList(requireFlagValue(argv, i, '--distill-source-langs'));
      i += 1;
      continue;
    }
    if (arg === '--distill-target-langs') {
      parsed.distillTargetLangs = parseStringList(requireFlagValue(argv, i, '--distill-target-langs'));
      i += 1;
      continue;
    }
    if (arg === '--distill-pair-allowlist') {
      parsed.distillPairAllowlist = parseStringList(requireFlagValue(argv, i, '--distill-pair-allowlist'));
      i += 1;
      continue;
    }
    if (arg === '--strict-pair-contract') {
      parsed.strictPairContract = true;
      continue;
    }
    if (arg === '--distill-shard-index') {
      parsed.distillShardIndex = parsePositiveInteger(requireFlagValue(argv, i, '--distill-shard-index'), '--distill-shard-index');
      i += 1;
      continue;
    }
    if (arg === '--distill-shard-count') {
      parsed.distillShardCount = parsePositiveInteger(requireFlagValue(argv, i, '--distill-shard-count'), '--distill-shard-count');
      i += 1;
      continue;
    }
    if (arg === '--resume-from') {
      parsed.resumeFrom = parseOptionalString(requireFlagValue(argv, i, '--resume-from'));
      i += 1;
      continue;
    }
    if (arg === '--force-resume') {
      parsed.forceResume = true;
      continue;
    }
    if (arg === '--force-resume-reason') {
      parsed.forceResumeReason = parseOptionalString(requireFlagValue(argv, i, '--force-resume-reason'));
      i += 1;
      continue;
    }
    if (arg === '--force-resume-source') {
      parsed.forceResumeSource = parseOptionalString(requireFlagValue(argv, i, '--force-resume-source'));
      i += 1;
      continue;
    }
    if (arg === '--checkpoint-operator') {
      parsed.checkpointOperator = parseOptionalString(requireFlagValue(argv, i, '--checkpoint-operator'));
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
      parsed.stageAArtifact = parseOptionalString(requireFlagValue(argv, i, '--stage-a-artifact'));
      i += 1;
      continue;
    }
    if (arg === '--stage-a-artifact-hash') {
      parsed.stageAArtifactHash = parseOptionalString(requireFlagValue(argv, i, '--stage-a-artifact-hash'));
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
    tiny: 'distill-translategemma-tiny',
    medium: 'distill-translategemma-medium',
  };
  const candidate = presets[workloadArg] || workloadArg;
  const loaded = await loadTrainingWorkloadPack(candidate, {
    registryPath: 'tools/configs/training-workloads/registry.json',
  });
  if (loaded.workload.kind !== 'distill') {
    throw new Error(`Expected distill workload, got "${loaded.workload.kind}" from ${loaded.path}`);
  }
  return {
    ...loaded,
    config: JSON.parse(loaded.raw),
  };
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

  if (!forceResumeReason || !forceResumeSource) {
    throw new Error('forceResume=true requires explicit forceResumeReason and forceResumeSource.');
  }
  return {
    forceResume: true,
    forceResumeReason,
    forceResumeSource,
    checkpointOperator,
  };
}

function findStagePlanEntry(workload, stageId) {
  const entries = Array.isArray(workload?.pipeline?.stagePlan) ? workload.pipeline.stagePlan : [];
  return entries.find((entry) => entry?.id === stageId || entry?.trainingStage === stageId) || null;
}

function deriveLanguagePair(workload) {
  const sourceLangs = Array.isArray(workload?.pipeline?.sourceLangs) ? workload.pipeline.sourceLangs : [];
  const targetLangs = Array.isArray(workload?.pipeline?.targetLangs) ? workload.pipeline.targetLangs : [];
  if (sourceLangs.length === 1 && targetLangs.length === 1) {
    return `${sourceLangs[0]}-${targetLangs[0]}`;
  }
  const pairAllowlist = Array.isArray(workload?.pipeline?.pairAllowlist) ? workload.pipeline.pairAllowlist : [];
  if (pairAllowlist.length === 1) {
    const match = /^([^-\s>]+)\s*->\s*([^-\s>]+)$/u.exec(String(pairAllowlist[0]));
    if (match) {
      return `${match[1]}-${match[2]}`;
    }
  }
  return null;
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

export function resolveDistillWorkloadOptions(args, loadedWorkload) {
  const workload = loadedWorkload.workload;
  const stageAPlan = findStagePlanEntry(workload, 'stage_a');
  const stageBPlan = findStagePlanEntry(workload, 'stage_b');
  if (!stageAPlan || !stageBPlan) {
    throw new Error(`Distill workload ${loadedWorkload.path} must define both stage_a and stage_b stagePlan entries.`);
  }
  const inferredBenchSteps = stageAPlan.steps === stageBPlan.steps ? stageAPlan.steps : null;
  const benchSteps = args.trainingBenchSteps
    ?? inferredBenchSteps;
  if (args.mode === 'bench' && benchSteps == null) {
    throw new Error(
      `Distill workload ${loadedWorkload.path} has divergent stage_a/stage_b steps; pass --training-bench-steps explicitly.`
    );
  }
  const distillDatasetPath = args.distillDatasetPath || workload.datasetPath;
  const distillLanguagePair = deriveLanguagePair(workload);
  if (!distillDatasetPath) {
    throw new Error(`Distill workload ${loadedWorkload.path} is missing datasetPath.`);
  }
  if (!distillLanguagePair) {
    throw new Error(
      `Distill workload ${loadedWorkload.path} must resolve a language pair from pipeline.sourceLangs/targetLangs or pairAllowlist.`
    );
  }
  return {
    trainingSchemaVersion: workload.trainingSchemaVersion,
    benchSteps,
    stageASteps: args.stageASteps ?? stageAPlan.steps,
    stageBSteps: args.stageBSteps ?? stageBPlan.steps,
    checkpointEvery: args.checkpointEvery ?? workload.checkpointEvery,
    teacherModelId: workload.teacherModelId,
    studentModelId: workload.studentModelId,
    distillDatasetId: workload.datasetId,
    distillDatasetPath,
    distillLanguagePair,
    distillSourceLangs: args.distillSourceLangs || workload.pipeline.sourceLangs || null,
    distillTargetLangs: args.distillTargetLangs || workload.pipeline.targetLangs || null,
    distillPairAllowlist: args.distillPairAllowlist || workload.pipeline.pairAllowlist || null,
    strictPairContract: args.strictPairContract || workload.pipeline.strictPairContract === true,
    runtimeConfigJson: buildRuntimeConfigJson(loadedWorkload.config.benchRun),
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
  const args = [
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
    '--json',
  ];
  if (options.runtimeConfigJson !== null) {
    args.splice(args.length - 1, 0, '--runtime-config', options.runtimeConfigJson);
  }
  return runCli(args);
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const outDirAbs = resolve(args.outDir);
  const workload = await loadWorkloadConfig(args.workload);
  const resolvedWorkload = resolveDistillWorkloadOptions(args, workload);
  const trainingSchemaVersion = Number(resolvedWorkload.trainingSchemaVersion);
  const benchSteps = resolvedWorkload.benchSteps;
  const stageASteps = resolvedWorkload.stageASteps;
  const stageBSteps = resolvedWorkload.stageBSteps;
  const checkpointEvery = parseOptionalPositiveInteger(
    resolvedWorkload.checkpointEvery,
    'checkpointEvery'
  );
  const teacherModelId = String(resolvedWorkload.teacherModelId);
  const studentModelId = String(resolvedWorkload.studentModelId);
  const distillDatasetId = String(resolvedWorkload.distillDatasetId);
  const distillDatasetPath = resolvedWorkload.distillDatasetPath;
  const distillSourceLangs = resolvedWorkload.distillSourceLangs;
  const distillTargetLangs = resolvedWorkload.distillTargetLangs;
  const distillPairAllowlist = resolvedWorkload.distillPairAllowlist;
  const strictPairContract = resolvedWorkload.strictPairContract;
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
  const distillLanguagePair = String(resolvedWorkload.distillLanguagePair);
  const runtimeConfigJson = resolvedWorkload.runtimeConfigJson;
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

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
