import { initDevice, getKernelCapabilities } from '../gpu/device.js';
import { setPlatformsBaseUrl } from '../config/platforms/loader.js';
import { setRegistryUrl } from '../config/kernels/registry.js';
import { createTrainingConfig } from '../config/training-defaults.js';
import { runMatmul } from '../gpu/kernels/index.js';
import { createTensor } from '../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../memory/buffer-pool.js';
import { OpType } from './autograd.js';
import { AdamOptimizer } from './optimizer.js';
import { TrainingRunner } from './runner.js';
import { trainStep } from './trainer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { sha256Hex } from '../utils/sha256.js';
import { computeSampleStats } from '../debug/stats.js';

const LEGACY_BROWSER_TESTS = Object.freeze([
  'loss-forward',
  'softmax-backward',
  'cross-entropy-backward',
  'rmsnorm-backward',
  'layernorm-backward',
  'conv2d-backward',
  'matmul-backward',
  'embed-backward',
  'ebm-state-optimize',
  'ebm-recorded-bench',
  'parity-fixture',
  'training-leak-perf',
  'autograd-branching',
]);
const TRAINING_COMMAND_SCHEMA_VERSION = 1;

function buildSuiteSummary(suiteName, results, startTimeMs) {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  for (const result of results) {
    if (result.skipped) {
      skipped++;
    } else if (result.passed) {
      passed++;
    } else {
      failed++;
    }
  }
  return {
    suite: suiteName,
    passed,
    failed,
    skipped,
    duration: Math.max(0, performance.now() - startTimeMs),
    results,
  };
}

function normalizeTrainingTestNames(names) {
  if (!Array.isArray(names)) return null;
  const normalized = names
    .map((name) => String(name || '').trim())
    .filter(Boolean);
  return normalized.length > 0 ? normalized : null;
}

function assertTrainingSchemaVersion(value) {
  if (value === undefined || value === null) {
    return TRAINING_COMMAND_SCHEMA_VERSION;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed !== TRAINING_COMMAND_SCHEMA_VERSION) {
    throw new Error(`trainingSchemaVersion must be ${TRAINING_COMMAND_SCHEMA_VERSION}.`);
  }
  return parsed;
}

function makeTensorFromFloat32(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor');
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label || 'train_tensor');
}

function makeTensorFromUint32(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tokens');
  uploadData(buffer, data);
  // Token tensors are wrapped as f32 by contract; kernels read the underlying u32 bytes.
  return createTensor(buffer, 'f32', shape, label || 'train_tokens');
}

function releaseTensor(tensor) {
  if (!tensor?.buffer) return;
  releaseBuffer(tensor.buffer);
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function resolveRuntimeUrl(pathname) {
  if (typeof globalThis.location !== 'undefined' && globalThis.location?.href) {
    return pathname;
  }
  return new URL(pathname, import.meta.url).toString();
}

async function ensureTrainingGpuRuntime() {
  setPlatformsBaseUrl(resolveRuntimeUrl('../config/platforms/'));
  setRegistryUrl(resolveRuntimeUrl('../config/kernels/registry.json'));
  await initDevice();
}

function createToyModelFixture(overrides = {}) {
  const config = createTrainingConfig({
    ...overrides,
    training: {
      enabled: true,
      lossScaling: { enabled: false },
      gradient: { maxNorm: 0 },
      ...(overrides.training || {}),
    },
  });

  const encoderWeight = makeTensorFromFloat32(
    [0.1, -0.2, 0.3, 0.4, 0.05, -0.1],
    [3, 2],
    'training_suite_encoder_weight'
  );
  const priorWeight = makeTensorFromFloat32(
    [0.02, -0.01, 0.03, -0.05, 0.04, -0.02],
    [3, 2],
    'training_suite_prior_weight'
  );
  const decoderWeight = makeTensorFromFloat32(
    [0.03, 0.02, -0.01, 0.06, -0.04, 0.02],
    [3, 2],
    'training_suite_decoder_weight'
  );
  const baseWeight = makeTensorFromFloat32(
    [0.08, -0.12, 0.16, 0.22, -0.03, 0.09],
    [3, 2],
    'training_suite_base_weight'
  );
  const input = makeTensorFromFloat32([0.5, 0.1, -0.3, 0.2, 0.4, -0.1], [2, 3], 'training_suite_input');
  const targets = makeTensorFromUint32([1, 0], [2], 'training_suite_targets');
  const batch = { input, targets };

  const model = {
    async forward(inputTensor, tape) {
      return tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, 2, 2, 3, { transposeB: false }),
        [inputTensor, baseWeight],
        { M: 2, N: 2, K: 3, transposeB: false }
      );
    },
    loraParams() {
      return [baseWeight];
    },
    paramGroups() {
      return {
        encoder: [encoderWeight],
        prior: [priorWeight],
        decoder: [decoderWeight],
        base: [baseWeight],
        lora: [baseWeight],
      };
    },
  };

  return {
    config,
    model,
    batch,
    cleanup() {
      releaseTensor(encoderWeight);
      releaseTensor(priorWeight);
      releaseTensor(decoderWeight);
      releaseTensor(baseWeight);
      releaseTensor(input);
      releaseTensor(targets);
    },
  };
}

async function runRunnerSmokeTest() {
  const fixture = createToyModelFixture();
  try {
    const runner = new TrainingRunner(fixture.config, {
      optimizer: new AdamOptimizer(fixture.config),
      crossEntropyLoss,
      clipGradients,
    });
    const dataset = {
      async *batches() {
        for (let i = 0; i < 3; i += 1) {
          yield fixture.batch;
        }
      },
    };

    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: 1,
      shuffle: false,
      maxSteps: 3,
    });
    if (!Array.isArray(metrics) || metrics.length === 0) {
      return { passed: false, error: 'Training runner produced no metrics.' };
    }
    for (const entry of metrics) {
      if (!isFiniteNumber(entry.total_loss) || !isFiniteNumber(entry.step_time_ms)) {
        return { passed: false, error: 'Training runner emitted non-finite metrics.' };
      }
    }

    return { passed: true };
  } finally {
    fixture.cleanup();
  }
}

async function runTrainStepMetricsTest() {
  const fixture = createToyModelFixture();
  try {
    const result = await trainStep(fixture.model, fixture.batch, fixture.config, {
      crossEntropyLoss,
      clipGradients,
      optimizer: new AdamOptimizer(fixture.config),
    });

    if (!isFiniteNumber(result.forward_ms) || !isFiniteNumber(result.backward_ms)) {
      return { passed: false, error: 'trainStep did not report finite phase timings.' };
    }
    if (!result.clipMetrics || !isFiniteNumber(result.clipMetrics.gradient_norm_unclipped)) {
      return { passed: false, error: 'trainStep did not report clipping metrics.' };
    }
    if (!result.optimizerMetrics || !isFiniteNumber(result.optimizerMetrics.optimizer_ms)) {
      return { passed: false, error: 'trainStep did not report optimizer metrics.' };
    }

    return { passed: true };
  } finally {
    fixture.cleanup();
  }
}

const UL_STAGE_SET = Object.freeze(['stage1_joint', 'stage2_base']);
const DISTILL_STAGE_SET = Object.freeze(['stage_a', 'stage_b']);
const TRAINING_STAGE_SET = Object.freeze([...UL_STAGE_SET, ...DISTILL_STAGE_SET]);

function normalizeTrainingStage(stage) {
  const normalized = String(stage || '').trim();
  if (!normalized) return null;
  if (!TRAINING_STAGE_SET.includes(normalized)) {
    throw new Error(`Unknown training stage "${normalized}". Expected one of: ${TRAINING_STAGE_SET.join(', ')}.`);
  }
  return normalized;
}

function isUlStage(stage) {
  return UL_STAGE_SET.includes(String(stage || ''));
}

function isDistillStage(stage) {
  return DISTILL_STAGE_SET.includes(String(stage || ''));
}

function normalizeTrainingConfigOverride(value) {
  if (!value) return null;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('trainingConfig must be an object when provided.');
  }
  return value;
}

function buildUlTrainingOverrides(options = {}) {
  const trainingConfig = normalizeTrainingConfigOverride(options.trainingConfig);
  const explicitStage = normalizeTrainingStage(options.trainingStage || trainingConfig?.ul?.stage);
  const ulEnabled = isUlStage(explicitStage) || trainingConfig?.ul?.enabled === true;
  if (!ulEnabled) {
    return trainingConfig || null;
  }
  const stage = isUlStage(explicitStage) ? explicitStage : 'stage1_joint';
  const ulOverride = {
    ...(trainingConfig?.ul || {}),
    enabled: true,
    stage,
    stage1Artifact: options.stage1Artifact ?? trainingConfig?.ul?.stage1Artifact ?? null,
    stage1ArtifactHash: options.stage1ArtifactHash ?? trainingConfig?.ul?.stage1ArtifactHash ?? null,
    artifactDir: options.ulArtifactDir ?? trainingConfig?.ul?.artifactDir ?? 'bench/out/ul',
  };
  if (stage === 'stage2_base') {
    ulOverride.freeze = {
      encoder: true,
      prior: true,
      decoder: true,
      base: false,
      lora: false,
      ...(trainingConfig?.ul?.freeze || {}),
    };
  }
  return {
    ...(trainingConfig || {}),
    ul: ulOverride,
  };
}

function buildDistillTrainingOverrides(options = {}) {
  const trainingConfig = normalizeTrainingConfigOverride(options.trainingConfig);
  const explicitStage = normalizeTrainingStage(options.trainingStage || trainingConfig?.distill?.stage);
  const distillEnabled = isDistillStage(explicitStage) || trainingConfig?.distill?.enabled === true;
  if (!distillEnabled) {
    return trainingConfig || null;
  }
  const stage = isDistillStage(explicitStage) ? explicitStage : 'stage_a';
  const distillOverride = {
    ...(trainingConfig?.distill || {}),
    enabled: true,
    stage,
    teacherModelId: options.teacherModelId ?? trainingConfig?.distill?.teacherModelId ?? null,
    studentModelId: options.studentModelId ?? trainingConfig?.distill?.studentModelId ?? null,
    datasetId: options.distillDatasetId ?? trainingConfig?.distill?.datasetId ?? null,
    languagePair: options.distillLanguagePair ?? trainingConfig?.distill?.languagePair ?? null,
    stageAArtifact: options.stageAArtifact ?? trainingConfig?.distill?.stageAArtifact ?? null,
    stageAArtifactHash: options.stageAArtifactHash ?? trainingConfig?.distill?.stageAArtifactHash ?? null,
    artifactDir: options.distillArtifactDir ?? trainingConfig?.distill?.artifactDir ?? 'bench/out/distill',
  };
  if (stage === 'stage_b') {
    distillOverride.freeze = {
      encoder: true,
      prior: true,
      decoder: true,
      base: false,
      lora: false,
      ...(trainingConfig?.distill?.freeze || {}),
    };
  }
  return {
    ...(trainingConfig || {}),
    distill: distillOverride,
  };
}

async function computeNodeFileHash(filePath) {
  if (!(typeof process !== 'undefined' && process.versions?.node)) {
    return null;
  }
  const [{ readFile }, { resolve }] = await Promise.all([
    import('node:fs/promises'),
    import('node:path'),
  ]);
  const absolutePath = resolve(String(filePath));
  const raw = await readFile(absolutePath, 'utf8');
  return {
    absolutePath,
    hash: sha256Hex(raw),
  };
}

async function runUlStageTest(stage, options = {}) {
  const ulTraining = buildUlTrainingOverrides({
    ...options,
    trainingStage: stage,
  });
  const fixture = createToyModelFixture({
    training: ulTraining || undefined,
  });

  try {
    const runner = new TrainingRunner(fixture.config, {
      optimizer: new AdamOptimizer(fixture.config),
      crossEntropyLoss,
      clipGradients,
    });
    const dataset = {
      async *batches() {
        for (let i = 0; i < 2; i += 1) {
          yield fixture.batch;
        }
      },
    };
    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: 1,
      shuffle: false,
      maxSteps: 2,
      modelId: options.modelId || 'training',
      modelUrl: options.modelUrl || null,
      timestamp: options.timestamp || null,
      ulArtifactDir: options.ulArtifactDir || null,
    });
    if (!Array.isArray(metrics) || metrics.length === 0) {
      return { passed: false, error: `UL ${stage} produced no metrics.` };
    }
    const requiredFields = [
      'loss_prior',
      'loss_decoder',
      'loss_recon',
      'lambda',
      'latent_bitrate_proxy',
      'loss_total',
      'coeff_ce',
      'coeff_prior',
      'coeff_decoder',
      'coeff_recon',
    ];
    if (stage === 'stage1_joint') {
      requiredFields.push(
        'schedule_step_index',
        'latent_clean_mean',
        'latent_clean_std',
        'latent_noise_mean',
        'latent_noise_std',
        'latent_noisy_mean',
        'latent_noisy_std',
        'latent_shape',
        'latent_clean_values',
        'latent_noise_values',
        'latent_noisy_values'
      );
    }
    if (stage === 'stage2_base') {
      requiredFields.push('stage1_latent_count');
    }
    for (const field of requiredFields) {
      if (!(field in metrics[0])) {
        return { passed: false, error: `UL ${stage} missing metric field "${field}".` };
      }
    }
    const artifact = runner.lastArtifact;
    if (!artifact || !artifact.manifestPath) {
      return { passed: false, error: `UL ${stage} did not produce artifacts.` };
    }
    return {
      passed: true,
      artifact,
      metrics: {
        stage,
        steps: metrics.length,
        manifestPath: artifact.manifestPath,
        manifestHash: artifact.manifestHash,
        manifestContentHash: artifact.manifestContentHash,
        manifestFileHash: artifact.manifestFileHash ?? null,
        ulResolvedConfig: {
          enabled: fixture.config.training?.ul?.enabled === true,
          stage: fixture.config.training?.ul?.stage ?? null,
          lambda0: fixture.config.training?.ul?.lambda0 ?? null,
          seed: fixture.config.training?.ul?.seed ?? null,
          noiseSchedule: fixture.config.training?.ul?.noiseSchedule ?? null,
          priorAlignment: fixture.config.training?.ul?.priorAlignment ?? null,
          decoderSigmoidWeight: fixture.config.training?.ul?.decoderSigmoidWeight ?? null,
          freeze: fixture.config.training?.ul?.freeze ?? null,
        },
      },
    };
  } finally {
    fixture.cleanup();
  }
}

async function runUlStage1Test(options = {}) {
  return runUlStageTest('stage1_joint', options);
}

async function runUlStage2Test(options = {}) {
  const explicitStage1Artifact = String(options.stage1Artifact || '').trim();
  let stage1Artifact = explicitStage1Artifact || null;
  let stage1ArtifactHash = String(options.stage1ArtifactHash || '').trim() || null;

  if (!stage1Artifact) {
    const stage1 = await runUlStage1Test({
      ...options,
      trainingStage: 'stage1_joint',
    });
    if (!stage1?.passed || !stage1?.artifact?.manifestPath) {
      return { passed: false, error: 'UL stage2 preflight failed to generate stage1 artifact.' };
    }
    stage1Artifact = stage1.artifact.manifestPath;
    stage1ArtifactHash = stage1.artifact.manifestHash;
    const nodeHash = await computeNodeFileHash(stage1Artifact);
    if (nodeHash?.hash) {
      stage1ArtifactHash = nodeHash.hash;
      stage1Artifact = nodeHash.absolutePath;
    }
  }

  return runUlStageTest('stage2_base', {
    ...options,
    stage1Artifact,
    stage1ArtifactHash,
  });
}

async function runDistillStageTest(stage, options = {}) {
  const distillTraining = buildDistillTrainingOverrides({
    ...options,
    trainingStage: stage,
  });
  const fixture = createToyModelFixture({
    training: distillTraining || undefined,
  });

  try {
    const runner = new TrainingRunner(fixture.config, {
      optimizer: new AdamOptimizer(fixture.config),
      crossEntropyLoss,
      clipGradients,
    });
    const dataset = {
      async *batches() {
        for (let i = 0; i < 2; i += 1) {
          yield fixture.batch;
        }
      },
    };
    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: 1,
      shuffle: false,
      maxSteps: 2,
      modelId: options.modelId || 'training',
      modelUrl: options.modelUrl || null,
      timestamp: options.timestamp || null,
      distillArtifactDir: options.distillArtifactDir || null,
      stageAArtifact: options.stageAArtifact || null,
      stageAArtifactHash: options.stageAArtifactHash || null,
      teacherModelId: options.teacherModelId || null,
      studentModelId: options.studentModelId || null,
      distillDatasetId: options.distillDatasetId || null,
      distillLanguagePair: options.distillLanguagePair || null,
    });
    if (!Array.isArray(metrics) || metrics.length === 0) {
      return { passed: false, error: `Distill ${stage} produced no metrics.` };
    }
    const requiredFields = stage === 'stage_a'
      ? ['loss_kd', 'distill_stage']
      : ['loss_triplet', 'distill_stage', 'distill_triplet_margin'];
    for (const field of requiredFields) {
      if (!(field in metrics[0])) {
        return { passed: false, error: `Distill ${stage} missing metric field "${field}".` };
      }
    }
    const artifact = runner.lastArtifact;
    if (!artifact || !artifact.manifestPath) {
      return { passed: false, error: `Distill ${stage} did not produce artifacts.` };
    }
    return {
      passed: true,
      artifact,
      metrics: {
        stage,
        steps: metrics.length,
        manifestPath: artifact.manifestPath,
        manifestHash: artifact.manifestHash,
        manifestContentHash: artifact.manifestContentHash,
        manifestFileHash: artifact.manifestFileHash ?? null,
        distillResolvedConfig: {
          enabled: fixture.config.training?.distill?.enabled === true,
          stage: fixture.config.training?.distill?.stage ?? null,
          teacherModelId: fixture.config.training?.distill?.teacherModelId ?? null,
          studentModelId: fixture.config.training?.distill?.studentModelId ?? null,
          datasetId: fixture.config.training?.distill?.datasetId ?? null,
          languagePair: fixture.config.training?.distill?.languagePair ?? null,
          temperature: fixture.config.training?.distill?.temperature ?? null,
          alphaKd: fixture.config.training?.distill?.alphaKd ?? null,
          alphaCe: fixture.config.training?.distill?.alphaCe ?? null,
          tripletMargin: fixture.config.training?.distill?.tripletMargin ?? null,
          freeze: fixture.config.training?.distill?.freeze ?? null,
        },
      },
    };
  } finally {
    fixture.cleanup();
  }
}

async function runDistillStageATest(options = {}) {
  return runDistillStageTest('stage_a', options);
}

async function runDistillStageBTest(options = {}) {
  const explicitStageAArtifact = String(options.stageAArtifact || '').trim();
  let stageAArtifact = explicitStageAArtifact || null;
  let stageAArtifactHash = String(options.stageAArtifactHash || '').trim() || null;

  if (!stageAArtifact) {
    const stageA = await runDistillStageATest({
      ...options,
      trainingStage: 'stage_a',
    });
    if (!stageA?.passed || !stageA?.artifact?.manifestPath) {
      return { passed: false, error: 'Distill stage_b preflight failed to generate stage_a artifact.' };
    }
    stageAArtifact = stageA.artifact.manifestPath;
    stageAArtifactHash = stageA.artifact.manifestHash;
    const nodeHash = await computeNodeFileHash(stageAArtifact);
    if (nodeHash?.hash) {
      stageAArtifactHash = nodeHash.hash;
      stageAArtifact = nodeHash.absolutePath;
    }
  }

  return runDistillStageTest('stage_b', {
    ...options,
    stageAArtifact,
    stageAArtifactHash,
  });
}

function createLegacySkippedTest(name) {
  return async () => ({
    passed: true,
    skipped: true,
    error: `Legacy browser-only test "${name}" remains in tests/training/browser/test-page.js.`,
  });
}

const CORE_TESTS = Object.freeze({
  'runner-smoke': runRunnerSmokeTest,
  'train-step-metrics': runTrainStepMetricsTest,
  'ul-stage1': runUlStage1Test,
  'ul-stage2': runUlStage2Test,
  'distill-stage-a': runDistillStageATest,
  'distill-stage-b': runDistillStageBTest,
});

const TESTS = Object.freeze({
  ...CORE_TESTS,
  ...Object.fromEntries(LEGACY_BROWSER_TESTS.map((name) => [name, createLegacySkippedTest(name)])),
});

export const trainingHarness = Object.freeze({
  async getGPU() {
    await ensureTrainingGpuRuntime();
    return true;
  },
  async runTest(name, options = {}) {
    const fn = TESTS[name];
    if (!fn) {
      return { passed: false, error: `Unknown training test: ${name}` };
    }
    return fn(options);
  },
  listTests() {
    return Object.keys(TESTS);
  },
});

export async function runTrainingSuite(options = {}) {
  const trainingSchemaVersion = assertTrainingSchemaVersion(options.trainingSchemaVersion);
  const startTime = performance.now();
  await trainingHarness.getGPU();

  const availableTests = trainingHarness.listTests();
  const requestedTestsFromOptions = normalizeTrainingTestNames(options.trainingTests);
  const requestedStage = normalizeTrainingStage(options.trainingStage);
  const stageDefaultTests = requestedStage === 'stage1_joint'
    ? ['ul-stage1']
    : (
      requestedStage === 'stage2_base'
        ? ['ul-stage2']
        : (
          requestedStage === 'stage_a'
            ? ['distill-stage-a']
            : (requestedStage === 'stage_b' ? ['distill-stage-b'] : null)
        )
    );
  const requestedTests = requestedTestsFromOptions || stageDefaultTests;
  if (requestedTests) {
    const unknownTests = requestedTests.filter((name) => !availableTests.includes(name));
    if (unknownTests.length > 0) {
      throw new Error(`Unknown training test(s): ${unknownTests.join(', ')}`);
    }
  }
  const testsToRun = requestedTests ?? availableTests;

  const results = [];
  for (const testName of testsToRun) {
    const testStart = performance.now();
    try {
      const outcome = await trainingHarness.runTest(testName, options);
      const passed = outcome?.passed === true;
      const skipped = outcome?.skipped === true;
      const errorMessage = skipped
        ? (outcome?.error ? String(outcome.error) : undefined)
        : (passed ? undefined : String(outcome?.error || 'Training test failed'));
      const entry = {
        name: testName,
        passed,
        skipped,
        duration: Math.max(0, performance.now() - testStart),
        ...(errorMessage ? { error: errorMessage } : {}),
      };
      if (outcome?.metrics && typeof outcome.metrics === 'object') {
        entry.metrics = outcome.metrics;
      }
      if (outcome?.artifact && typeof outcome.artifact === 'object') {
        entry.artifact = outcome.artifact;
      }
      results.push(entry);
    } catch (error) {
      results.push({
        name: testName,
        passed: false,
        duration: Math.max(0, performance.now() - testStart),
        error: String(error?.message || error),
      });
    }
  }

  const summary = buildSuiteSummary('training', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || options.modelUrl || 'training',
    metrics: {
      testsRun: results.length,
      selectedTests: testsToRun,
      availableTests,
      trainingStage: requestedStage || null,
      trainingSchemaVersion,
    },
    deviceInfo: getKernelCapabilities(),
  };
}

function toPositiveInteger(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const floored = Math.floor(parsed);
  return floored > 0 ? floored : fallback;
}

function resolveBenchRunSettings(options = {}) {
  const benchRun = options.benchRun && typeof options.benchRun === 'object'
    ? options.benchRun
    : {};
  return {
    warmupRuns: Math.max(0, Math.floor(Number(benchRun.warmupRuns) || 0)),
    timedRuns: toPositiveInteger(benchRun.timedRuns, 1),
    stepsPerRun: toPositiveInteger(
      options.trainingBenchSteps ?? benchRun.steps ?? options.trainingSteps,
      2
    ),
  };
}

function resolveTrainingOverrides(options = {}) {
  const distillTraining = buildDistillTrainingOverrides(options);
  if (distillTraining?.distill?.enabled) {
    return distillTraining;
  }
  const ulTraining = buildUlTrainingOverrides(options);
  if (ulTraining) {
    return ulTraining;
  }
  return normalizeTrainingConfigOverride(options.trainingConfig) || undefined;
}

export async function runTrainingBenchSuite(options = {}) {
  const trainingSchemaVersion = assertTrainingSchemaVersion(options.trainingSchemaVersion);
  const startTime = performance.now();
  await trainingHarness.getGPU();

  const benchSettings = resolveBenchRunSettings(options);
  const totalRuns = benchSettings.warmupRuns + benchSettings.timedRuns;
  const trainingOverrides = resolveTrainingOverrides(options);

  const timedRunDurationsMs = [];
  const timedRunStepsPerSec = [];
  const timedStepDurationsMs = [];
  const timedRunUlArtifacts = [];
  const timedRunDistillArtifacts = [];
  const trainingMetricsReport = [];
  let completedTimedRuns = 0;

  for (let runIndex = 0; runIndex < totalRuns; runIndex += 1) {
    const fixture = createToyModelFixture({
      training: trainingOverrides,
    });
    try {
      const runner = new TrainingRunner(fixture.config, {
        optimizer: new AdamOptimizer(fixture.config),
        crossEntropyLoss,
        clipGradients,
      });
      const dataset = {
        async *batches() {
          for (let i = 0; i < benchSettings.stepsPerRun; i += 1) {
            yield fixture.batch;
          }
        },
      };

      const runStart = performance.now();
      const runMetrics = await runner.run(fixture.model, dataset, {
        epochs: 1,
        batchSize: 1,
        shuffle: false,
        maxSteps: benchSettings.stepsPerRun,
        modelId: options.modelId || 'training',
        modelUrl: options.modelUrl || null,
        timestamp: options.timestamp || null,
        ulArtifactDir: options.ulArtifactDir || null,
        distillArtifactDir: options.distillArtifactDir || null,
        stageAArtifact: options.stageAArtifact || null,
        stageAArtifactHash: options.stageAArtifactHash || null,
        teacherModelId: options.teacherModelId || null,
        studentModelId: options.studentModelId || null,
        distillDatasetId: options.distillDatasetId || null,
        distillLanguagePair: options.distillLanguagePair || null,
      });
      const runDurationMs = Math.max(0, performance.now() - runStart);
      const isTimedRun = runIndex >= benchSettings.warmupRuns;
      if (isTimedRun) {
        completedTimedRuns += 1;
        timedRunDurationsMs.push(runDurationMs);
        const runStepCount = Array.isArray(runMetrics) ? runMetrics.length : 0;
        if (runDurationMs > 0 && runStepCount > 0) {
          timedRunStepsPerSec.push((runStepCount * 1000) / runDurationMs);
        }
        for (const stepEntry of runMetrics) {
          if (isFiniteNumber(stepEntry?.step_time_ms)) {
            timedStepDurationsMs.push(stepEntry.step_time_ms);
          }
          trainingMetricsReport.push(stepEntry);
        }
        if (runner.lastArtifact && typeof runner.lastArtifact === 'object') {
          const artifactEntry = {
            runIndex: completedTimedRuns,
            ...runner.lastArtifact,
          };
          if (runner.lastArtifact.kind === 'distill') {
            timedRunDistillArtifacts.push(artifactEntry);
          } else {
            timedRunUlArtifacts.push(artifactEntry);
          }
        }
      }
    } finally {
      fixture.cleanup();
    }
  }

  const runMsStats = computeSampleStats(timedRunDurationsMs);
  const stepMsStats = computeSampleStats(timedStepDurationsMs);
  const stepsPerSecStats = computeSampleStats(timedRunStepsPerSec);

  const results = [
    {
      name: 'training-benchmark',
      passed: completedTimedRuns > 0 && trainingMetricsReport.length > 0,
      duration: Math.max(0, performance.now() - startTime),
      error: completedTimedRuns > 0 && trainingMetricsReport.length > 0
        ? undefined
        : 'No timed training benchmark runs completed.',
    },
  ];

  const summary = buildSuiteSummary('bench', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || options.modelUrl || 'training',
    metrics: {
      workloadType: 'training',
      warmupRuns: benchSettings.warmupRuns,
      timedRuns: benchSettings.timedRuns,
      completedTimedRuns,
      stepsPerRun: benchSettings.stepsPerRun,
      trainingSchemaVersion,
      trainingMetricsReport,
      ulArtifacts: timedRunUlArtifacts,
      distillArtifacts: timedRunDistillArtifacts,
      latency: {
        runMs: runMsStats,
        stepMs: stepMsStats,
      },
      throughput: {
        stepsPerSec: stepsPerSecStats,
      },
    },
    deviceInfo: getKernelCapabilities(),
  };
}
