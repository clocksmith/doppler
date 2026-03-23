import { isPlainObject } from '../utils/plain-object.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import {
  TOOLING_COMMAND_SET,
  TOOLING_INTENT_SET,
  TOOLING_WORKLOAD_SET,
  TRAINING_STAGE_SET,
} from './command-api-constants.js';

export function asOptionalString(value, label) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value !== 'string') {
    throw new Error(`tooling command: ${label} must be a string when provided.`);
  }
  const trimmed = value.trim();
  return trimmed || null;
}

export function asOptionalBoolean(value, label) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'boolean') {
    throw new Error(`tooling command: ${label} must be a boolean when provided.`);
  }
  return value;
}

export function asOptionalObject(value, label) {
  if (value === undefined || value === null) return null;
  if (!isPlainObject(value)) {
    throw new Error(`tooling command: ${label} must be an object when provided.`);
  }
  return value;
}

export function asOptionalStringArray(value, label) {
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

export function asOptionalPositiveInteger(value, label) {
  if (value === undefined || value === null || value === '') return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`tooling command: ${label} must be a positive integer when provided.`);
  }
  return parsed;
}

export function asOptionalTrainingStage(value, label) {
  const stage = asOptionalString(value, label);
  if (!stage) return null;
  if (!TRAINING_STAGE_SET.includes(stage)) {
    throw new Error(`tooling command: ${label} must be one of ${TRAINING_STAGE_SET.join(', ')}.`);
  }
  return stage;
}

export function asOptionalForceResumeReason(value, label) {
  const reason = asOptionalString(value, label);
  if (!reason) return null;
  return reason;
}

export function asOptionalAction(value, label, allowed) {
  const action = asOptionalString(value, label);
  if (!action) return null;
  if (!allowed.includes(action)) {
    throw new Error(`tooling command: ${label} must be one of ${allowed.join(', ')}.`);
  }
  return action;
}

export function assertCommand(value) {
  const command = asOptionalString(value, 'command');
  if (!command) {
    throw new Error('tooling command: command is required.');
  }
  if (!TOOLING_COMMAND_SET.includes(command)) {
    throw new Error(`tooling command: unsupported command "${command}".`);
  }
  return command;
}

export function resolveCommandRuntimeContract(command) {
  const runtimeContract = selectRuleValue('tooling', 'commandRuntime', 'runtimeContract', { command });
  if (!isPlainObject(runtimeContract)) {
    throw new Error(`tooling command: missing runtime contract metadata for "${command}".`);
  }

  const workload = runtimeContract.workload == null
    ? null
    : asOptionalString(runtimeContract.workload, `runtime contract workload for "${command}"`);
  if (workload && !TOOLING_WORKLOAD_SET.includes(workload)) {
    throw new Error(`tooling command: runtime contract workload "${workload}" is not supported.`);
  }

  const intent = runtimeContract.intent == null
    ? null
    : asOptionalString(runtimeContract.intent, `runtime contract intent for "${command}"`);
  if (intent && !TOOLING_INTENT_SET.includes(intent)) {
    throw new Error(`tooling command: runtime contract intent "${intent}" is not supported.`);
  }

  return {
    workload,
    intent,
  };
}

export function asOptionalCacheMode(value, label) {
  const cacheMode = asOptionalString(value, label);
  if (!cacheMode) return null;
  if (cacheMode !== 'cold' && cacheMode !== 'warm') {
    throw new Error(`${label} must be "cold" or "warm"`);
  }
  return cacheMode;
}

export function asOptionalLoadMode(value, label) {
  const loadMode = asOptionalString(value, label);
  if (!loadMode) return null;
  if (loadMode !== 'opfs' && loadMode !== 'http' && loadMode !== 'memory') {
    throw new Error(`${label} must be "opfs", "http", or "memory"`);
  }
  return loadMode;
}

export function assertModelId(value, command, workload) {
  const modelId = asOptionalString(value, 'modelId');
  if (!modelId) {
    throw new Error(
      `tooling command: modelId is required for command "${command}" (workload "${workload}").`
    );
  }
  return modelId;
}

export function assertForbiddenStringField(raw, fieldName, command) {
  const value = asOptionalString(raw[fieldName], fieldName);
  if (value) {
    throw new Error(
      `tooling command: ${command} does not accept ${fieldName}.`
    );
  }
}

export function assertForbiddenObjectField(raw, fieldName, command) {
  const value = asOptionalObject(raw[fieldName], fieldName);
  if (value) {
    throw new Error(
      `tooling command: ${command} does not accept ${fieldName}.`
    );
  }
}

export function assertForbiddenConfigChainField(raw, command) {
  const value = asOptionalStringArray(raw.configChain, 'configChain');
  if (value) {
    throw new Error(
      `tooling command: ${command} does not accept configChain. ` +
      'configChain is only supported by harness/browser-manifest runtime composition.'
    );
  }
}

export function resolveWorkloadForCommand(raw, command, runtimeContract) {
  const inputWorkload = asOptionalString(raw.workload, 'workload')
    ?? asOptionalString(raw.suite, 'suite');
  if (runtimeContract.workload) {
    if (inputWorkload && inputWorkload !== runtimeContract.workload) {
      throw new Error(
        `tooling command: "${command}" requires workload "${runtimeContract.workload}" and does not accept "${inputWorkload}".`
      );
    }
    return runtimeContract.workload;
  }

  const workload = inputWorkload;
  if (!workload) {
    throw new Error(`tooling command: workload is required for "${command}".`);
  }
  if (!TOOLING_WORKLOAD_SET.includes(workload)) {
    throw new Error(`tooling command: unsupported workload "${workload}".`);
  }
  return workload;
}

export function createCommandRequestBase(raw, command) {
  return {
    command,
    workload: null,
    intent: null,
    action: null,
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
    runtimeProfile: asOptionalString(raw.runtimeProfile, 'runtimeProfile'),
    runtimeConfigUrl: asOptionalString(raw.runtimeConfigUrl, 'runtimeConfigUrl'),
    runtimeConfig: asOptionalObject(raw.runtimeConfig, 'runtimeConfig'),
    inputDir: null,
    outputDir: null,
    convertPayload: null,
    workloadPath: null,
    runRoot: null,
    checkpointPath: null,
    checkpointId: null,
    checkpointStep: null,
    stageId: null,
    stageArtifact: null,
    subsetManifest: null,
    evalDatasetId: null,
    pollIntervalMs: null,
    stopWhenIdle: null,
    captureOutput: false,
    keepPipeline: false,
    report: asOptionalObject(raw.report, 'report'),
    timestamp: raw.timestamp ?? null,
    searchParams: raw.searchParams ?? null,
    baselineProvider: asOptionalString(raw.baselineProvider, 'baselineProvider'),
    observedProvider: asOptionalString(raw.observedProvider, 'observedProvider'),
  };
}
