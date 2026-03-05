import { isPlainObject } from '../utils/plain-object.js';
import { selectRuleValue } from '../rules/rule-registry.js';

const TOOLING_COMMAND_SET = ['convert', 'debug', 'bench', 'verify'];
const TOOLING_SURFACE_SET = ['browser', 'node'];
const TOOLING_SUITE_SET = ['kernels', 'inference', 'training', 'bench', 'debug', 'diffusion', 'energy'];
const TOOLING_INTENT_SET = ['verify', 'investigate', 'calibrate'];
const VERIFY_SUITES = ['kernels', 'inference', 'training', 'diffusion', 'energy'];
const TRAINING_STAGE_SET = ['stage1_joint', 'stage2_base', 'stage_a', 'stage_b'];
const TRAINING_COMMAND_SCHEMA_VERSION = 1;

export const TOOLING_COMMANDS = Object.freeze([...TOOLING_COMMAND_SET]);
export const TOOLING_SURFACES = Object.freeze([...TOOLING_SURFACE_SET]);
export const TOOLING_SUITES = Object.freeze([...TOOLING_SUITE_SET]);
export const TOOLING_VERIFY_SUITES = Object.freeze([...VERIFY_SUITES]);
export const TOOLING_TRAINING_COMMAND_SCHEMA_VERSION = TRAINING_COMMAND_SCHEMA_VERSION;

function asOptionalString(value, label) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value !== 'string') {
    throw new Error(`tooling command: ${label} must be a string when provided.`);
  }
  const trimmed = value.trim();
  return trimmed || null;
}

function asOptionalBoolean(value, label) {
  if (value === undefined || value === null) return null;
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

function asOptionalForceResumeReason(value, label) {
  const reason = asOptionalString(value, label);
  if (!reason) return null;
  return reason;
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

function resolveCommandRuntimeContract(command) {
  const runtimeContract = selectRuleValue('tooling', 'commandRuntime', 'runtimeContract', { command });
  if (!isPlainObject(runtimeContract)) {
    throw new Error(`tooling command: missing runtime contract metadata for "${command}".`);
  }

  const suite = runtimeContract.suite == null
    ? null
    : asOptionalString(runtimeContract.suite, `runtime contract suite for "${command}"`);
  if (suite && !TOOLING_SUITE_SET.includes(suite)) {
    throw new Error(`tooling command: runtime contract suite "${suite}" is not supported.`);
  }

  const intent = runtimeContract.intent == null
    ? null
    : asOptionalString(runtimeContract.intent, `runtime contract intent for "${command}"`);
  if (intent && !TOOLING_INTENT_SET.includes(intent)) {
    throw new Error(`tooling command: runtime contract intent "${intent}" is not supported.`);
  }

  return {
    suite,
    intent,
  };
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
    useGpuCast: asOptionalBoolean(
      execution.useGpuCast,
      'convertPayload.execution.useGpuCast'
    ),
    gpuCastMinTensorBytes: asOptionalPositiveInteger(
      execution.gpuCastMinTensorBytes,
      'convertPayload.execution.gpuCastMinTensorBytes'
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
    distillDatasetPath: null,
    distillLanguagePair: null,
    distillSourceLangs: null,
    distillTargetLangs: null,
    distillPairAllowlist: null,
    strictPairContract: null,
    distillShardIndex: null,
    distillShardCount: null,
    resumeFrom: null,
    forceResume: null,
    forceResumeReason: null,
    forceResumeSource: null,
    checkpointOperator: null,
    trainingSchemaVersion: null,
    trainingBenchSteps: null,
    checkpointEvery: null,
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
  const runtimeContract = resolveCommandRuntimeContract(command);
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
  const distillDatasetPath = asOptionalString(raw.distillDatasetPath, 'distillDatasetPath');
  const distillLanguagePair = asOptionalString(raw.distillLanguagePair, 'distillLanguagePair');
  const distillSourceLangs = asOptionalStringArray(raw.distillSourceLangs, 'distillSourceLangs');
  const distillTargetLangs = asOptionalStringArray(raw.distillTargetLangs, 'distillTargetLangs');
  const distillPairAllowlist = asOptionalStringArray(raw.distillPairAllowlist, 'distillPairAllowlist');
  const strictPairContract = asOptionalBoolean(raw.strictPairContract, 'strictPairContract');
  const distillShardIndex = asOptionalPositiveInteger(raw.distillShardIndex, 'distillShardIndex');
  const distillShardCount = asOptionalPositiveInteger(raw.distillShardCount, 'distillShardCount');
  const resumeFrom = asOptionalString(raw.resumeFrom, 'resumeFrom');
  const forceResume = asOptionalBoolean(raw.forceResume, 'forceResume');
  const forceResumeReason = asOptionalForceResumeReason(raw.forceResumeReason, 'forceResumeReason');
  const forceResumeSource = asOptionalString(raw.forceResumeSource, 'forceResumeSource');
  const checkpointOperator = asOptionalString(raw.checkpointOperator, 'checkpointOperator');
  const trainingSchemaVersionInput = asOptionalPositiveInteger(
    raw.trainingSchemaVersion,
    'trainingSchemaVersion'
  );
  const trainingBenchSteps = asOptionalPositiveInteger(raw.trainingBenchSteps, 'trainingBenchSteps');
  const checkpointEvery = asOptionalPositiveInteger(raw.checkpointEvery, 'checkpointEvery');
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
    || distillDatasetPath
    || distillLanguagePair
    || distillSourceLangs
    || distillTargetLangs
    || distillPairAllowlist
    || strictPairContract !== null
    || distillShardIndex
    || distillShardCount
    || resumeFrom
    || forceResume !== null
    || forceResumeReason
    || forceResumeSource
    || checkpointOperator
    || trainingSchemaVersionInput
    || trainingBenchSteps
    || checkpointEvery
  )) {
    throw new Error(
      'tooling command: training-only fields require suite="training" or bench workloadType="training".'
    );
  }
  if (forceResumeReason && forceResume !== true) {
    throw new Error(
      'tooling command: forceResumeReason requires forceResume=true.'
    );
  }
  if (forceResumeSource && forceResume !== true) {
    throw new Error(
      'tooling command: forceResumeSource requires forceResume=true.'
    );
  }
  if (checkpointOperator && forceResume !== true) {
    throw new Error(
      'tooling command: checkpointOperator requires forceResume=true.'
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
  if (
    distillShardIndex != null
    && distillShardCount != null
    && distillShardIndex > distillShardCount
  ) {
    throw new Error('tooling command: distillShardIndex must be <= distillShardCount.');
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
    || !!trainingConfig?.distill?.datasetPath
    || !!teacherModelId
    || !!studentModelId
    || !!distillDatasetPath
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
    distillDatasetPath,
    distillLanguagePair,
    distillSourceLangs,
    distillTargetLangs,
    distillPairAllowlist,
    strictPairContract: allowsTrainingFields ? strictPairContract : null,
    distillShardIndex,
    distillShardCount,
    resumeFrom,
    forceResume: allowsTrainingFields
      ? (forceResume == null ? null : forceResume === true)
      : null,
    forceResumeReason: allowsTrainingFields ? forceResumeReason : null,
    forceResumeSource: allowsTrainingFields ? forceResumeSource : null,
    checkpointOperator: allowsTrainingFields ? checkpointOperator : null,
    trainingSchemaVersion,
    trainingBenchSteps,
    checkpointEvery: allowsTrainingFields ? checkpointEvery : null,
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
