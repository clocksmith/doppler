import { isPlainObject } from '../utils/plain-object.js';
import { cloneJsonValue as cloneCommandValue } from '../utils/clone-json.js';
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


function normalizeInferencePrompt(value, label) {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      throw new Error(`tooling command: ${label} must not be empty when provided.`);
    }
    return trimmed;
  }
  if (Array.isArray(value) || isPlainObject(value)) {
    return cloneCommandValue(value);
  }
  throw new Error(`tooling command: ${label} must be a string, array, or object when provided.`);
}

function normalizeInferenceImagePixels(value, label) {
  if (value === undefined || value === null) {
    return null;
  }
  const bytes = ArrayBuffer.isView(value)
    ? Array.from(new Uint8Array(value.buffer, value.byteOffset, value.byteLength))
    : (Array.isArray(value) ? value : null);
  if (!Array.isArray(bytes)) {
    throw new Error(`tooling command: ${label} must be an array or typed array when provided.`);
  }
  return bytes.map((entry, index) => {
    const parsed = Number(entry);
    if (!Number.isInteger(parsed) || parsed < 0 || parsed > 255) {
      throw new Error(`tooling command: ${label}[${index}] must be an integer in [0, 255].`);
    }
    return parsed;
  });
}

function normalizeInferenceImage(value, label) {
  const image = asOptionalObject(value, label);
  if (!image) {
    return null;
  }

  const allowedKeys = new Set(['url', 'width', 'height', 'pixels', 'pixelDataBase64']);
  for (const key of Object.keys(image)) {
    if (!allowedKeys.has(key)) {
      throw new Error(`tooling command: ${label}.${key} is not supported.`);
    }
  }

  const url = asOptionalString(image.url, `${label}.url`);
  const width = asOptionalPositiveInteger(image.width, `${label}.width`);
  const height = asOptionalPositiveInteger(image.height, `${label}.height`);
  const pixelDataBase64 = asOptionalString(image.pixelDataBase64, `${label}.pixelDataBase64`);
  const pixels = normalizeInferenceImagePixels(image.pixels, `${label}.pixels`);
  const sourceCount = [url, pixelDataBase64, pixels].filter((entry) => entry != null).length;
  if (sourceCount !== 1) {
    throw new Error(
      `tooling command: ${label} requires exactly one source: url, pixelDataBase64, or pixels.`
    );
  }
  if (url) {
    if (width != null || height != null) {
      throw new Error(
        `tooling command: ${label}.width and ${label}.height are only valid for raw pixel inputs.`
      );
    }
    return {
      url,
      width: null,
      height: null,
      pixels: null,
      pixelDataBase64: null,
    };
  }
  if (width == null || height == null) {
    throw new Error(
      `tooling command: ${label}.width and ${label}.height are required for raw pixel inputs.`
    );
  }
  return {
    url: null,
    width,
    height,
    pixels,
    pixelDataBase64,
  };
}

function normalizeInferenceInput(value, workload) {
  const inferenceInput = asOptionalObject(value, 'inferenceInput');
  if (!inferenceInput) {
    return null;
  }
  if (workload !== 'inference') {
    throw new Error('tooling command: inferenceInput requires workload="inference".');
  }

  const allowedKeys = new Set(['prompt', 'image', 'maxTokens', 'softTokenBudget']);
  for (const key of Object.keys(inferenceInput)) {
    if (!allowedKeys.has(key)) {
      throw new Error(`tooling command: inferenceInput.${key} is not supported.`);
    }
  }

  const prompt = normalizeInferencePrompt(inferenceInput.prompt, 'inferenceInput.prompt');
  const image = normalizeInferenceImage(inferenceInput.image, 'inferenceInput.image');
  const maxTokens = asOptionalPositiveInteger(inferenceInput.maxTokens, 'inferenceInput.maxTokens');
  const softTokenBudget = asOptionalPositiveInteger(
    inferenceInput.softTokenBudget,
    'inferenceInput.softTokenBudget'
  );

  if (softTokenBudget != null && image == null) {
    throw new Error('tooling command: inferenceInput.softTokenBudget requires inferenceInput.image.');
  }
  if (image && prompt != null && typeof prompt !== 'string') {
    throw new Error(
      'tooling command: inferenceInput.prompt must be a string when inferenceInput.image is provided.'
    );
  }
  if (prompt == null && image == null && maxTokens == null && softTokenBudget == null) {
    throw new Error(
      'tooling command: inferenceInput must specify prompt, image, maxTokens, or softTokenBudget.'
    );
  }

  return {
    prompt,
    image,
    maxTokens,
    softTokenBudget,
  };
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
  const inferenceInput = normalizeInferenceInput(raw.inferenceInput, workload);
  const inputWorkloadType = asOptionalString(raw.workloadType, 'workloadType');
  const programBundle = asOptionalObject(raw.programBundle, 'programBundle');
  const programBundlePath = asOptionalString(raw.programBundlePath, 'programBundlePath');
  const parityProviders = asOptionalStringArray(raw.parityProviders, 'parityProviders');
  const programBundleParityMode = asOptionalString(raw.programBundleParityMode, 'programBundleParityMode');
  if (programBundleParityMode && programBundleParityMode !== 'contract' && programBundleParityMode !== 'execute') {
    throw new Error('tooling command: programBundleParityMode must be "contract" or "execute".');
  }
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
  const isProgramBundleParity = command === 'verify'
    && workload === 'inference'
    && workloadType === 'program-bundle';
  if (isProgramBundleParity) {
    if (!programBundle && !programBundlePath) {
      throw new Error('tooling command: program-bundle parity requires programBundle or programBundlePath.');
    }
  } else if (programBundle || programBundlePath || parityProviders || programBundleParityMode) {
    throw new Error(
      'tooling command: programBundle, programBundlePath, parityProviders, and programBundleParityMode require ' +
      'command="verify", workload="inference", and workloadType="program-bundle".'
    );
  }
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

  const requiresModel = workload !== 'kernels' && workload !== 'training' && !isProgramBundleParity;
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
    inferenceInput,
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
    programBundle: isProgramBundleParity ? programBundle : null,
    programBundlePath: isProgramBundleParity ? programBundlePath : null,
    parityProviders: isProgramBundleParity ? parityProviders : null,
    programBundleParityMode: isProgramBundleParity ? (programBundleParityMode ?? 'contract') : null,
    modelUrl,
    captureOutput: asOptionalBoolean(raw.captureOutput, 'captureOutput') ?? false,
    keepPipeline: asOptionalBoolean(raw.keepPipeline, 'keepPipeline') ?? false,
  };
}
