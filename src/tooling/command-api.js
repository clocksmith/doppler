import { isPlainObject } from '../utils/plain-object.js';

const TOOLING_COMMAND_SET = ['convert', 'debug', 'bench', 'test-model'];
const TOOLING_SURFACE_SET = ['browser', 'node'];
const TOOLING_SUITE_SET = ['kernels', 'inference', 'training', 'bench', 'debug', 'diffusion', 'energy'];
const VERIFY_SUITES = ['kernels', 'inference', 'training', 'diffusion', 'energy'];
const TRAINING_STAGE_SET = ['stage1_joint', 'stage2_base', 'stage_a', 'stage_b'];
const TRAINING_COMMAND_SCHEMA_VERSION = 1;

export const TOOLING_COMMANDS = Object.freeze([...TOOLING_COMMAND_SET]);
export const TOOLING_SURFACES = Object.freeze([...TOOLING_SURFACE_SET]);
export const TOOLING_SUITES = Object.freeze([...TOOLING_SUITE_SET]);
export const TOOLING_VERIFY_SUITES = Object.freeze([...VERIFY_SUITES]);
export const TOOLING_TRAINING_COMMAND_SCHEMA_VERSION = TRAINING_COMMAND_SCHEMA_VERSION;

const COMMAND_RUNTIME_CONTRACT = Object.freeze({
  debug: Object.freeze({ suite: 'debug', intent: 'investigate' }),
  bench: Object.freeze({ suite: 'bench', intent: 'calibrate' }),
  'test-model': Object.freeze({ suite: null, intent: 'verify' }),
});

function asOptionalString(value, label) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value !== 'string') {
    throw new Error(`tooling command: ${label} must be a string when provided.`);
  }
  const trimmed = value.trim();
  return trimmed || null;
}

function asOptionalBoolean(value, label) {
  if (value === undefined) return null;
  if (typeof value !== 'boolean') {
    throw new Error(`tooling command: ${label} must be a boolean when provided.`);
  }
  return value;
}

function asOptionalObject(value, label) {
  if (value === undefined || value === null) return null;
  if (!isPlainObject(value)) {
    throw new Error(`tooling command: ${label} must be an object when provided.`);
  }
  return value;
}

function asOptionalStringArray(value, label) {
  if (value === undefined || value === null) return null;
  if (!Array.isArray(value)) {
    throw new Error(`tooling command: ${label} must be an array of strings when provided.`);
  }
  const normalized = value.map((entry, index) => {
    if (typeof entry !== 'string') {
      throw new Error(`tooling command: ${label}[${index}] must be a string.`);
    }
    const trimmed = entry.trim();
    if (!trimmed) {
      throw new Error(`tooling command: ${label}[${index}] must not be empty.`);
    }
    return trimmed;
  });
  return normalized.length > 0 ? normalized : null;
}

function asOptionalPositiveInteger(value, label) {
  if (value === undefined || value === null || value === '') return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`tooling command: ${label} must be a positive integer when provided.`);
  }
  return parsed;
}

function asOptionalTrainingStage(value, label) {
  const stage = asOptionalString(value, label);
  if (!stage) return null;
  if (!TRAINING_STAGE_SET.includes(stage)) {
    throw new Error(`tooling command: ${label} must be one of ${TRAINING_STAGE_SET.join(', ')}.`);
  }
  return stage;
}

function assertCommand(value) {
  const command = asOptionalString(value, 'command');
  if (!command) {
    throw new Error('tooling command: command is required.');
  }
  if (!TOOLING_COMMAND_SET.includes(command)) {
    throw new Error(`tooling command: unsupported command "${command}".`);
  }
  return command;
}

function assertSuite(value, command) {
  const suite = asOptionalString(value, 'suite');
  if (!suite) {
    throw new Error(`tooling command: suite is required for "${command}".`);
  }
  if (!TOOLING_SUITE_SET.includes(suite)) {
    throw new Error(`tooling command: unsupported suite "${suite}".`);
  }
  return suite;
}

function asOptionalCacheMode(value, label) {
  const cacheMode = asOptionalString(value, label);
  if (!cacheMode) return null;
  if (cacheMode !== 'cold' && cacheMode !== 'warm') {
    throw new Error(`${label} must be "cold" or "warm"`);
  }
  return cacheMode;
}

function asOptionalLoadMode(value, label) {
  const loadMode = asOptionalString(value, label);
  if (!loadMode) return null;
  if (loadMode !== 'opfs' && loadMode !== 'http' && loadMode !== 'memory') {
    throw new Error(`${label} must be "opfs", "http", or "memory"`);
  }
  return loadMode;
}

function assertModelId(value, command, suite) {
  const modelId = asOptionalString(value, 'modelId');
  if (!modelId) {
    throw new Error(
      `tooling command: modelId is required for command "${command}" (suite "${suite}").`
    );
  }
  return modelId;
}

function normalizeConvertExecution(value) {
  const execution = asOptionalObject(value, 'convertPayload.execution');
  if (!execution) return null;

  const workerCountPolicy = asOptionalString(
    execution.workerCountPolicy,
    'convertPayload.execution.workerCountPolicy'
  );
  if (workerCountPolicy && workerCountPolicy !== 'cap' && workerCountPolicy !== 'error') {
    throw new Error(
      'tooling command: convertPayload.execution.workerCountPolicy must be "cap" or "error" when provided.'
    );
  }

  return {
    ...execution,
    workers: asOptionalPositiveInteger(
      execution.workers,
      'convertPayload.execution.workers'
    ),
    workerCountPolicy,
    maxInFlightJobs: asOptionalPositiveInteger(
      execution.maxInFlightJobs,
      'convertPayload.execution.maxInFlightJobs'
    ),
    rowChunkRows: asOptionalPositiveInteger(
      execution.rowChunkRows,
      'convertPayload.execution.rowChunkRows'
    ),
    rowChunkMinTensorBytes: asOptionalPositiveInteger(
      execution.rowChunkMinTensorBytes,
      'convertPayload.execution.rowChunkMinTensorBytes'
    ),
  };
}

function normalizeConvertPayload(value) {
  const payload = asOptionalObject(value, 'convertPayload');
  if (!payload) {
    throw new Error(
      'tooling command: convert requires convertPayload.converterConfig.'
    );
  }
  const converterConfig = asOptionalObject(
    payload.converterConfig,
    'convertPayload.converterConfig'
  );
  if (!converterConfig) {
    throw new Error(
      'tooling command: convert requires convertPayload.converterConfig.'
    );
  }
  return {
    ...payload,
    converterConfig,
    execution: normalizeConvertExecution(payload.execution),
  };
}

function normalizeConvert(raw) {
  const inputDir = asOptionalString(raw.inputDir, 'inputDir');
  const outputDir = asOptionalString(raw.outputDir, 'outputDir');
  const modelId = asOptionalString(raw.modelId, 'modelId');
  const payload = normalizeConvertPayload(raw.convertPayload);

  if (!inputDir) {
    throw new Error(
      'tooling command: convert requires inputDir.'
    );
  }
  if (modelId) {
    throw new Error(
      'tooling command: convert does not accept modelId. Set convertPayload.converterConfig.output.modelBaseId.'
    );
  }

  return {
    command: 'convert',
    suite: null,
    intent: null,
    modelId: null,
    trainingTests: null,
    trainingStage: null,
    trainingConfig: null,
    stage1Artifact: null,
    stage1ArtifactHash: null,
    ulArtifactDir: null,
    stageAArtifact: null,
    stageAArtifactHash: null,
    distillArtifactDir: null,
    teacherModelId: null,
    studentModelId: null,
    distillDatasetId: null,
    distillLanguagePair: null,
    trainingSchemaVersion: null,
    trainingBenchSteps: null,
    workloadType: asOptionalString(raw.workloadType, 'workloadType'),
    modelUrl: asOptionalString(raw.modelUrl, 'modelUrl'),
    cacheMode: asOptionalCacheMode(raw.cacheMode, 'cacheMode'),
    loadMode: asOptionalLoadMode(raw.loadMode, 'loadMode'),
    runtimePreset: asOptionalString(raw.runtimePreset, 'runtimePreset'),
    runtimeConfigUrl: asOptionalString(raw.runtimeConfigUrl, 'runtimeConfigUrl'),
    runtimeConfig: asOptionalObject(raw.runtimeConfig, 'runtimeConfig'),
    inputDir,
    outputDir,
    convertPayload: payload,
    captureOutput: false,
    keepPipeline: false,
    report: asOptionalObject(raw.report, 'report'),
    timestamp: raw.timestamp ?? null,
    searchParams: raw.searchParams ?? null,
  };
}

function normalizeSuiteCommand(raw, command) {
  const runtimeContract = COMMAND_RUNTIME_CONTRACT[command];
  let suite = runtimeContract.suite;
  if (!suite) {
    suite = assertSuite(raw.suite, command);
    if (!VERIFY_SUITES.includes(suite)) {
      throw new Error(
        `tooling command: "${command}" suite must be one of ${VERIFY_SUITES.join(', ')}.`
      );
    }
  }

  const modelUrl = asOptionalString(raw.modelUrl, 'modelUrl');
  const trainingTests = asOptionalStringArray(raw.trainingTests, 'trainingTests');
  const trainingStage = asOptionalTrainingStage(raw.trainingStage, 'trainingStage');
  const trainingConfig = asOptionalObject(raw.trainingConfig, 'trainingConfig');
  const stage1Artifact = asOptionalString(raw.stage1Artifact, 'stage1Artifact');
  const stage1ArtifactHash = asOptionalString(raw.stage1ArtifactHash, 'stage1ArtifactHash');
  const ulArtifactDir = asOptionalString(raw.ulArtifactDir, 'ulArtifactDir');
  const stageAArtifact = asOptionalString(raw.stageAArtifact, 'stageAArtifact');
  const stageAArtifactHash = asOptionalString(raw.stageAArtifactHash, 'stageAArtifactHash');
  const distillArtifactDir = asOptionalString(raw.distillArtifactDir, 'distillArtifactDir');
  const teacherModelId = asOptionalString(raw.teacherModelId, 'teacherModelId');
  const studentModelId = asOptionalString(raw.studentModelId, 'studentModelId');
  const distillDatasetId = asOptionalString(raw.distillDatasetId, 'distillDatasetId');
  const distillLanguagePair = asOptionalString(raw.distillLanguagePair, 'distillLanguagePair');
  const trainingSchemaVersionInput = asOptionalPositiveInteger(
    raw.trainingSchemaVersion,
    'trainingSchemaVersion'
  );
  const trainingBenchSteps = asOptionalPositiveInteger(raw.trainingBenchSteps, 'trainingBenchSteps');
  const workloadType = asOptionalString(raw.workloadType, 'workloadType');
  const isTrainingBenchWorkload = command === 'bench' && suite === 'bench' && workloadType === 'training';
  const allowsTrainingFields = suite === 'training' || isTrainingBenchWorkload;
  if (!allowsTrainingFields && (
    trainingTests
    || trainingStage
    || trainingConfig
    || stage1Artifact
    || stage1ArtifactHash
    || ulArtifactDir
    || stageAArtifact
    || stageAArtifactHash
    || distillArtifactDir
    || teacherModelId
    || studentModelId
    || distillDatasetId
    || distillLanguagePair
    || trainingSchemaVersionInput
    || trainingBenchSteps
  )) {
    throw new Error(
      'tooling command: training-only fields require suite="training" or bench workloadType="training".'
    );
  }
  const trainingSchemaVersion = allowsTrainingFields
    ? (trainingSchemaVersionInput ?? TRAINING_COMMAND_SCHEMA_VERSION)
    : null;
  if (trainingSchemaVersionInput != null && trainingSchemaVersionInput !== TRAINING_COMMAND_SCHEMA_VERSION) {
    throw new Error(
      `tooling command: trainingSchemaVersion must be ${TRAINING_COMMAND_SCHEMA_VERSION}.`
    );
  }

  const requiresModel = suite !== 'kernels' && !isTrainingBenchWorkload;
  const hasTrainingSource = allowsTrainingFields && (
    !!modelUrl
    || !!trainingStage
    || !!stage1Artifact
    || !!stageAArtifact
    || !!trainingConfig?.ul?.stage
    || !!trainingConfig?.distill?.stage
    || !!trainingConfig?.dataset
    || !!trainingConfig?.distill?.datasetId
    || !!teacherModelId
    || !!studentModelId
  );
  const modelId = (requiresModel && !hasTrainingSource)
    ? assertModelId(raw.modelId, command, suite)
    : asOptionalString(raw.modelId, 'modelId');

  return {
    command,
    suite,
    intent: runtimeContract.intent,
    modelId,
    trainingTests,
    trainingStage,
    trainingConfig,
    stage1Artifact,
    stage1ArtifactHash,
    ulArtifactDir,
    stageAArtifact,
    stageAArtifactHash,
    distillArtifactDir,
    teacherModelId,
    studentModelId,
    distillDatasetId,
    distillLanguagePair,
    trainingSchemaVersion,
    trainingBenchSteps,
    workloadType,
    modelUrl,
    cacheMode: asOptionalCacheMode(raw.cacheMode, 'cacheMode'),
    loadMode: asOptionalLoadMode(raw.loadMode, 'loadMode'),
    runtimePreset: asOptionalString(raw.runtimePreset, 'runtimePreset'),
    runtimeConfigUrl: asOptionalString(raw.runtimeConfigUrl, 'runtimeConfigUrl'),
    runtimeConfig: asOptionalObject(raw.runtimeConfig, 'runtimeConfig'),
    inputDir: null,
    outputDir: null,
    convertPayload: null,
    captureOutput: asOptionalBoolean(raw.captureOutput, 'captureOutput') ?? false,
    keepPipeline: asOptionalBoolean(raw.keepPipeline, 'keepPipeline') ?? false,
    report: asOptionalObject(raw.report, 'report'),
    timestamp: raw.timestamp ?? null,
    searchParams: raw.searchParams ?? null,
  };
}

export function normalizeToolingCommandRequest(input) {
  if (!isPlainObject(input)) {
    throw new Error('tooling command: request must be an object.');
  }
  const command = assertCommand(input.command);
  if (command === 'convert') {
    return normalizeConvert(input);
  }
  return normalizeSuiteCommand(input, command);
}

export function buildRuntimeContractPatch(commandRequest) {
  const request = normalizeToolingCommandRequest(commandRequest);
  if (!request.suite || !request.intent) {
    return null;
  }

  return {
    shared: {
      harness: {
        mode: request.suite,
        modelId: request.modelId ?? null,
      },
      tooling: {
        intent: request.intent,
      },
    },
  };
}

export function ensureCommandSupportedOnSurface(commandRequest, surface) {
  const request = normalizeToolingCommandRequest(commandRequest);
  const normalizedSurface = asOptionalString(surface, 'surface');
  if (!normalizedSurface || !TOOLING_SURFACE_SET.includes(normalizedSurface)) {
    throw new Error(`tooling command: unsupported surface "${surface}".`);
  }

  // All commands are contractually available on both surfaces.
  // Surface-specific capability checks happen in the runners.
  return {
    request,
    surface: normalizedSurface,
  };
}
