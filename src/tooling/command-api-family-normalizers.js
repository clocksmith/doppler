import {
  DISTILL_ACTION_SET,
  LORA_ACTION_SET,
  TRAINING_COMMAND_SCHEMA_VERSION,
  VERIFY_WORKLOADS,
} from './command-api-constants.js';
import {
  asOptionalAction,
  asOptionalBoolean,
  asOptionalForceResumeReason,
  asOptionalObject,
  asOptionalPositiveInteger,
  asOptionalString,
  asOptionalStringArray,
  asOptionalTrainingStage,
  assertForbiddenConfigChainField,
  assertForbiddenObjectField,
  assertForbiddenStringField,
  assertModelId,
  createCommandRequestBase,
  resolveCommandRuntimeContract,
  resolveWorkloadForCommand,
} from './command-api-helpers.js';

function resolveDebugRequestWorkload(raw) {
  const workload = asOptionalString(raw.workload, 'workload')
    ?? asOptionalString(raw.suite, 'suite');
  if (!workload) {
    return 'inference';
  }
  if (workload !== 'inference' && workload !== 'embedding') {
    throw new Error(
      'tooling command: "debug" workload must be "inference" or "embedding".'
    );
  }
  return workload;
}

function resolveBenchRequestWorkload(raw) {
  const workload = asOptionalString(raw.workload, 'workload')
    ?? asOptionalString(raw.suite, 'suite');
  if (!workload) {
    return 'inference';
  }
  if (
    workload !== 'inference'
    && workload !== 'embedding'
    && workload !== 'training'
    && workload !== 'diffusion'
  ) {
    throw new Error(
      'tooling command: "bench" workload must be "inference", "embedding", "training", or "diffusion".'
    );
  }
  return workload;
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

export function normalizeConvert(raw) {
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
  assertForbiddenStringField(raw, 'runtimeProfile', 'convert');
  assertForbiddenStringField(raw, 'runtimeConfigUrl', 'convert');
  assertForbiddenObjectField(raw, 'runtimeConfig', 'convert');
  assertForbiddenConfigChainField(raw, 'convert');

  return {
    ...createCommandRequestBase(raw, 'convert'),
    inputDir,
    outputDir,
    convertPayload: payload,
  };
}

export function normalizeTrainingOperatorCommand(raw, command) {
  assertForbiddenConfigChainField(raw, command);
  const allowedActions = command === 'distill' ? DISTILL_ACTION_SET : LORA_ACTION_SET;
  const action = asOptionalAction(raw.action, 'action', allowedActions);
  if (!action) {
    throw new Error(`tooling command: ${command} requires action.`);
  }
  const workloadPath = asOptionalString(raw.workloadPath, 'workloadPath');
  const runRoot = asOptionalString(raw.runRoot, 'runRoot');
  const checkpointPath = asOptionalString(raw.checkpointPath, 'checkpointPath');
  const checkpointId = asOptionalString(raw.checkpointId, 'checkpointId');
  const checkpointStep = asOptionalPositiveInteger(raw.checkpointStep, 'checkpointStep');
  const stageId = asOptionalString(raw.stageId, 'stageId');
  const stageArtifact = asOptionalString(raw.stageArtifact, 'stageArtifact');
  const subsetManifest = asOptionalString(raw.subsetManifest, 'subsetManifest');
  const evalDatasetId = asOptionalString(raw.evalDatasetId, 'evalDatasetId');
  const pollIntervalMs = asOptionalPositiveInteger(raw.pollIntervalMs, 'pollIntervalMs');
  const stopWhenIdle = asOptionalBoolean(raw.stopWhenIdle, 'stopWhenIdle');
  if (!workloadPath && !runRoot) {
    throw new Error(`tooling command: ${command} requires workloadPath or runRoot.`);
  }
  if ((action === 'eval' || action === 'export') && !checkpointPath && !runRoot) {
    throw new Error(`tooling command: ${command} ${action} requires checkpointPath or runRoot.`);
  }
  if (action === 'watch' && !runRoot) {
    throw new Error(`tooling command: ${command} watch requires runRoot.`);
  }
  if ((action === 'compare' || action === 'quality-gate') && !runRoot) {
    throw new Error(`tooling command: ${command} ${action} requires runRoot.`);
  }
  if (command === 'distill' && action === 'stage-b' && !stageArtifact && !runRoot) {
    throw new Error('tooling command: distill stage-b requires stageArtifact or runRoot.');
  }

  return {
    ...createCommandRequestBase(raw, command),
    action,
    workloadType: 'training',
    modelUrl: null,
    workloadPath,
    runRoot,
    checkpointPath,
    checkpointId,
    checkpointStep,
    stageId,
    stageArtifact,
    subsetManifest,
    evalDatasetId,
    pollIntervalMs,
    stopWhenIdle,
  };
}

export function normalizeSuiteCommand(raw, command) {
  assertForbiddenConfigChainField(raw, command);
  const runtimeContract = resolveCommandRuntimeContract(command);
  const workload = command === 'debug' || command === 'diagnose'
    ? resolveDebugRequestWorkload(raw)
    : (
      command === 'bench'
        ? resolveBenchRequestWorkload(raw)
        : resolveWorkloadForCommand(raw, command, runtimeContract)
    );
  if (!runtimeContract.workload && command === 'verify' && !VERIFY_WORKLOADS.includes(workload)) {
    throw new Error(
      `tooling command: "${command}" workload must be one of ${VERIFY_WORKLOADS.join(', ')}.`
    );
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
  const inputWorkloadType = asOptionalString(raw.workloadType, 'workloadType');
  if (
    inputWorkloadType
    && (workload === 'training' || workload === 'diffusion')
    && inputWorkloadType !== workload
  ) {
    throw new Error(
      `tooling command: workloadType "${inputWorkloadType}" does not match workload "${workload}".`
    );
  }
  const workloadType = inputWorkloadType ?? (
    command === 'bench' && (workload === 'training' || workload === 'diffusion')
      ? workload
      : null
  );
  const allowsTrainingFields = workload === 'training';
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
      'tooling command: training-only fields require workload="training".'
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

  const requiresModel = workload !== 'kernels' && workload !== 'training';
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
    ? assertModelId(raw.modelId, command, workload)
    : asOptionalString(raw.modelId, 'modelId');

  return {
    ...createCommandRequestBase(raw, command),
    workload,
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
    captureOutput: asOptionalBoolean(raw.captureOutput, 'captureOutput') ?? false,
    keepPipeline: asOptionalBoolean(raw.keepPipeline, 'keepPipeline') ?? false,
  };
}
