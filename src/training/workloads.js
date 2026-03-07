import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { isPlainObject } from '../utils/plain-object.js';
import { sha256Hex } from '../utils/sha256.js';
import { VALID_LORA_TARGET_MODULES } from '../config/schema/adapter.schema.js';

export const TRAINING_WORKLOAD_SCHEMA_VERSION = 1;
export const TRAINING_WORKLOAD_KINDS = Object.freeze(['lora', 'distill', 'ul']);
export const TRAINING_WORKLOAD_SURFACE_SUPPORT = Object.freeze(['node', 'browser', 'both']);
export const TRAINING_SELECTION_GOALS = Object.freeze(['max', 'min']);
export const TRAINING_EVAL_KINDS = Object.freeze([
  'translation',
  'text_generation',
  'classification',
  'retrieval',
  'custom',
]);

const LEGACY_DISTILL_TEST_IDS = Object.freeze(['distill-stage-a', 'distill-stage-b']);

function stableSortObject(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableSortObject(entry));
  }
  if (!isPlainObject(value)) {
    return value;
  }
  const sorted = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = stableSortObject(value[key]);
  }
  return sorted;
}

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function asNonEmptyString(value, label, options = {}) {
  if (value === undefined || value === null) {
    if (options.optional === true) return null;
    throw new Error(`${label} is required.`);
  }
  if (typeof value !== 'string') {
    throw new Error(`${label} must be a string.`);
  }
  const trimmed = value.trim();
  if (!trimmed) {
    if (options.optional === true) return null;
    throw new Error(`${label} must not be empty.`);
  }
  return trimmed;
}

function asPositiveInteger(value, label, options = {}) {
  if (value === undefined || value === null || value === '') {
    if (options.optional === true) return null;
    throw new Error(`${label} is required.`);
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function asFiniteNumber(value, label, options = {}) {
  if (value === undefined || value === null || value === '') {
    if (options.optional === true) return null;
    throw new Error(`${label} is required.`);
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return parsed;
}

function asBoolean(value, label, options = {}) {
  if (value === undefined || value === null) {
    if (options.optional === true) return null;
    throw new Error(`${label} is required.`);
  }
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be boolean.`);
  }
  return value;
}

function asStringArray(value, label, options = {}) {
  if (value === undefined || value === null) {
    if (options.optional === true) return null;
    throw new Error(`${label} is required.`);
  }
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array of strings.`);
  }
  const normalized = value.map((entry, index) => asNonEmptyString(entry, `${label}[${index}]`));
  if (normalized.length === 0 && options.allowEmpty !== true) {
    throw new Error(`${label} must not be empty.`);
  }
  return normalized;
}

function asEnum(value, label, allowed, options = {}) {
  const normalized = asNonEmptyString(value, label, options);
  if (normalized === null) return null;
  if (!allowed.includes(normalized)) {
    throw new Error(`${label} must be one of ${allowed.join(', ')}.`);
  }
  return normalized;
}

function asObject(value, label, options = {}) {
  if (value === undefined || value === null) {
    if (options.optional === true) return null;
    throw new Error(`${label} is required.`);
  }
  if (!isPlainObject(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function inferLegacyKind(payload, contextLabel) {
  const explicitKind = typeof payload?.kind === 'string' ? payload.kind.trim() : '';
  if (explicitKind) return explicitKind;
  const workloadKind = typeof payload?.workloadKind === 'string' ? payload.workloadKind.trim() : '';
  if (workloadKind) return workloadKind;
  const id = String(payload?.id || '').trim();
  if (id.startsWith('distill-')) return 'distill';
  if (id.startsWith('lora-')) return 'lora';
  if (id.startsWith('ul-') || id.startsWith('ul_training') || id.startsWith('ul-training')) {
    return 'ul';
  }
  if (Array.isArray(payload?.trainingTests) && payload.trainingTests.every((entry) => LEGACY_DISTILL_TEST_IDS.includes(String(entry)))) {
    return 'distill';
  }
  throw new Error(`${contextLabel}.kind is required.`);
}

function normalizeScheduler(value, label) {
  const scheduler = asObject(value, label, { optional: true }) || {};
  return {
    enabled: scheduler.enabled === true,
    type: asNonEmptyString(scheduler.type ?? 'constant', `${label}.type`),
    warmupSteps: asPositiveInteger(scheduler.warmupSteps ?? 0, `${label}.warmupSteps`, { optional: true }) ?? 0,
    stepSize: asPositiveInteger(scheduler.stepSize ?? 1, `${label}.stepSize`, { optional: true }) ?? 1,
    gamma: asFiniteNumber(scheduler.gamma ?? 1, `${label}.gamma`, { optional: true }) ?? 1,
    totalSteps: asPositiveInteger(scheduler.totalSteps ?? 1, `${label}.totalSteps`, { optional: true }) ?? 1,
    minLr: asFiniteNumber(scheduler.minLr ?? 0, `${label}.minLr`, { optional: true }) ?? 0,
  };
}

function normalizeTrainingConfig(value, label) {
  const training = asObject(value, label);
  const optimizer = asObject(training.optimizer, `${label}.optimizer`);
  const precision = isPlainObject(training.precision)
    ? training.precision
    : { activations: training.precision ?? 'f16' };
  const gradientClipping = isPlainObject(training.gradientClipping)
    ? training.gradientClipping
    : { maxNorm: training.gradientClipping ?? 1 };
  return {
    optimizer: {
      type: asNonEmptyString(optimizer.type ?? 'adam', `${label}.optimizer.type`),
      lr: asFiniteNumber(optimizer.lr, `${label}.optimizer.lr`),
      beta1: asFiniteNumber(optimizer.beta1 ?? 0.9, `${label}.optimizer.beta1`, { optional: true }) ?? 0.9,
      beta2: asFiniteNumber(optimizer.beta2 ?? 0.999, `${label}.optimizer.beta2`, { optional: true }) ?? 0.999,
      eps: asFiniteNumber(optimizer.eps ?? 1e-8, `${label}.optimizer.eps`, { optional: true }) ?? 1e-8,
      weightDecay: asFiniteNumber(optimizer.weightDecay ?? 0, `${label}.optimizer.weightDecay`, { optional: true }) ?? 0,
      scheduler: normalizeScheduler(optimizer.scheduler, `${label}.optimizer.scheduler`),
    },
    batchSize: asPositiveInteger(training.batchSize, `${label}.batchSize`),
    accumSteps: asPositiveInteger(training.accumSteps ?? 1, `${label}.accumSteps`, { optional: true }) ?? 1,
    steps: asPositiveInteger(training.steps, `${label}.steps`),
    precision: {
      activations: asNonEmptyString(precision.activations ?? 'f16', `${label}.precision.activations`),
      gradients: asNonEmptyString(precision.gradients ?? 'f32', `${label}.precision.gradients`),
      loraParams: asNonEmptyString(precision.loraParams ?? 'f32', `${label}.precision.loraParams`),
    },
    gradientClipping: {
      maxNorm: asFiniteNumber(gradientClipping.maxNorm ?? 1, `${label}.gradientClipping.maxNorm`, { optional: true }) ?? 1,
    },
  };
}

function normalizeEvalDatasets(value, label) {
  const entries = Array.isArray(value) ? value : [];
  return entries.map((entry, index) => {
    const dataset = asObject(entry, `${label}[${index}]`);
    return {
      id: asNonEmptyString(dataset.id, `${label}[${index}].id`),
      datasetPath: asNonEmptyString(dataset.datasetPath ?? dataset.path, `${label}[${index}].datasetPath`),
      evalKind: asEnum(
        dataset.evalKind ?? dataset.kind ?? 'text_generation',
        `${label}[${index}].evalKind`,
        TRAINING_EVAL_KINDS
      ),
      metrics: asStringArray(dataset.metrics ?? [], `${label}[${index}].metrics`, {
        optional: true,
        allowEmpty: true,
      }) ?? [],
      sourceLangs: asStringArray(dataset.sourceLangs, `${label}[${index}].sourceLangs`, { optional: true, allowEmpty: true }),
      targetLangs: asStringArray(dataset.targetLangs, `${label}[${index}].targetLangs`, { optional: true, allowEmpty: true }),
      pairAllowlist: asStringArray(dataset.pairAllowlist, `${label}[${index}].pairAllowlist`, { optional: true, allowEmpty: true }),
    };
  });
}

function normalizeFreezeConfig(value, label) {
  const freeze = asObject(value, label, { optional: true }) || {};
  return {
    encoder: freeze.encoder === true,
    prior: freeze.prior === true,
    decoder: freeze.decoder === true,
    base: freeze.base === true,
    lora: freeze.lora === true,
  };
}

function normalizeStagePlan(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty array.`);
  }
  return value.map((entry, index) => {
    const stage = asObject(entry, `${label}[${index}]`);
    const selectionMetric = asNonEmptyString(
      stage.selectionMetric ?? stage.metric ?? 'bleu',
      `${label}[${index}].selectionMetric`
    );
    const selectionGoal = asEnum(
      stage.selectionGoal ?? stage.goal ?? 'max',
      `${label}[${index}].selectionGoal`,
      TRAINING_SELECTION_GOALS
    );
    return {
      id: asNonEmptyString(stage.id ?? stage.name, `${label}[${index}].id`),
      trainingStage: asNonEmptyString(stage.trainingStage, `${label}[${index}].trainingStage`),
      objective: asNonEmptyString(stage.objective, `${label}[${index}].objective`),
      steps: asPositiveInteger(stage.steps, `${label}[${index}].steps`),
      checkpointEvery: asPositiveInteger(
        stage.checkpointEvery ?? stage.steps,
        `${label}[${index}].checkpointEvery`
      ),
      selectionMetric,
      selectionGoal,
      evalSchedule: asNonEmptyString(
        stage.evalSchedule ?? 'on_checkpoint',
        `${label}[${index}].evalSchedule`
      ),
    };
  });
}

function normalizeLoraConfig(value, label) {
  const lora = asObject(value, label);
  const adapter = asObject(lora.adapter, `${label}.adapter`);
  const exportConfig = asObject(lora.export, `${label}.export`, { optional: true });
  const activation = asObject(lora.activation, `${label}.activation`, { optional: true });
  const targetModules = asStringArray(adapter.targetModules, `${label}.adapter.targetModules`);
  for (const moduleName of targetModules) {
    if (!VALID_LORA_TARGET_MODULES.includes(moduleName)) {
      throw new Error(`${label}.adapter.targetModules contains unsupported module "${moduleName}".`);
    }
  }
  return {
    datasetFormat: asNonEmptyString(lora.datasetFormat ?? 'prompt_completion_jsonl', `${label}.datasetFormat`),
    taskType: asNonEmptyString(lora.taskType ?? 'text_generation', `${label}.taskType`),
    adapter: {
      rank: asPositiveInteger(adapter.rank, `${label}.adapter.rank`),
      alpha: asFiniteNumber(adapter.alpha, `${label}.adapter.alpha`),
      dropout: asFiniteNumber(adapter.dropout ?? 0, `${label}.adapter.dropout`, { optional: true }) ?? 0,
      targetModules,
    },
    freeze: normalizeFreezeConfig(lora.freeze, `${label}.freeze`),
    export: exportConfig
      ? {
        enabled: exportConfig.enabled !== false,
        atCheckpoints: exportConfig.atCheckpoints === true,
        select: asNonEmptyString(exportConfig.select ?? 'best', `${label}.export.select`),
        id: asNonEmptyString(exportConfig.id, `${label}.export.id`, { optional: true }),
        name: asNonEmptyString(exportConfig.name, `${label}.export.name`, { optional: true }),
        format: asNonEmptyString(exportConfig.format ?? 'manifest_json', `${label}.export.format`),
      }
      : null,
    activation: activation
      ? {
        enabled: activation.enabled === true,
        autoActivate: activation.autoActivate === true,
        smokePrompt: asNonEmptyString(activation.smokePrompt, `${label}.activation.smokePrompt`, { optional: true }),
      }
      : null,
  };
}

function normalizeDistillConfig(value, label) {
  const distill = asObject(value, label);
  return {
    stagePlan: normalizeStagePlan(distill.stagePlan, `${label}.stagePlan`),
    studentGraphMode: asNonEmptyString(
      distill.studentGraphMode ?? 'transformer_full',
      `${label}.studentGraphMode`
    ),
    temperature: asFiniteNumber(distill.temperature ?? 1, `${label}.temperature`, { optional: true }) ?? 1,
    alphaKd: asFiniteNumber(distill.alphaKd ?? 1, `${label}.alphaKd`, { optional: true }) ?? 1,
    alphaCe: asFiniteNumber(distill.alphaCe ?? 0, `${label}.alphaCe`, { optional: true }) ?? 0,
    tripletMargin: asFiniteNumber(distill.tripletMargin ?? 0.2, `${label}.tripletMargin`, { optional: true }) ?? 0.2,
    sourceLangs: asStringArray(distill.sourceLangs, `${label}.sourceLangs`, { optional: true, allowEmpty: true }),
    targetLangs: asStringArray(distill.targetLangs, `${label}.targetLangs`, { optional: true, allowEmpty: true }),
    pairAllowlist: asStringArray(distill.pairAllowlist, `${label}.pairAllowlist`, { optional: true, allowEmpty: true }),
    strictPairContract: asBoolean(
      distill.strictPairContract ?? false,
      `${label}.strictPairContract`,
      { optional: true }
    ) ?? false,
    subsetSpec: asObject(distill.subsetSpec, `${label}.subsetSpec`, { optional: true }),
  };
}

function normalizeLegacyUlPayload(payload, contextLabel) {
  return {
    schemaVersion: asPositiveInteger(payload.schemaVersion, `${contextLabel}.schemaVersion`),
    kind: 'ul',
    id: asNonEmptyString(payload.id, `${contextLabel}.id`),
    description: asNonEmptyString(payload.description, `${contextLabel}.description`),
    claimBoundary: asNonEmptyString(
      payload.claimBoundary ?? 'Practical UL workflow quality traceability.',
      `${contextLabel}.claimBoundary`
    ),
    seed: asPositiveInteger(payload.seed, `${contextLabel}.seed`),
    baseModelId: asNonEmptyString(payload.baseModelId ?? 'training', `${contextLabel}.baseModelId`),
    studentModelId: null,
    teacherModelId: null,
    datasetId: asNonEmptyString(payload.datasetId ?? payload.ulDatasetId ?? 'ul', `${contextLabel}.datasetId`),
    datasetPath: asNonEmptyString(payload.datasetPath ?? null, `${contextLabel}.datasetPath`, { optional: true }),
    evalDatasets: normalizeEvalDatasets(payload.evalDatasets ?? [], `${contextLabel}.evalDatasets`),
    trainingSchemaVersion: asPositiveInteger(
      payload.trainingSchemaVersion,
      `${contextLabel}.trainingSchemaVersion`
    ),
    checkpointEvery: asPositiveInteger(payload.checkpointEvery ?? 1, `${contextLabel}.checkpointEvery`),
    selectionMetric: asNonEmptyString(payload.selectionMetric ?? 'total_loss', `${contextLabel}.selectionMetric`),
    selectionGoal: asEnum(
      payload.selectionGoal ?? 'min',
      `${contextLabel}.selectionGoal`,
      TRAINING_SELECTION_GOALS
    ),
    surfaceSupport: asEnum(
      payload.surfaceSupport ?? 'node',
      `${contextLabel}.surfaceSupport`,
      TRAINING_WORKLOAD_SURFACE_SUPPORT
    ),
    training: normalizeTrainingConfig({
      optimizer: payload.training?.optimizer ?? {
        type: 'adam',
        lr: 2e-4,
      },
      batchSize: payload.training?.batchSize ?? 1,
      accumSteps: payload.training?.accumSteps ?? 1,
      steps: payload.training?.steps ?? payload.trainingBenchSteps ?? 1,
      precision: payload.training?.precision ?? {
        activations: 'f16',
        gradients: 'f32',
        loraParams: 'f32',
      },
      gradientClipping: payload.training?.gradientClipping ?? { maxNorm: 1 },
    }, `${contextLabel}.training`),
    pipeline: {
      legacyWorkloadType: 'ul',
      trainingTests: asStringArray(
        payload.trainingTests,
        `${contextLabel}.trainingTests`,
        { allowEmpty: false }
      ),
    },
  };
}

export function normalizeTrainingWorkloadPack(payload, context = {}) {
  const contextLabel = context.label || 'training workload';
  const kind = inferLegacyKind(payload, contextLabel);
  if (kind === 'ul') {
    const workload = normalizeLegacyUlPayload(payload, contextLabel);
    return withHashes(workload);
  }

  if (!TRAINING_WORKLOAD_KINDS.includes(kind)) {
    throw new Error(`${contextLabel}.kind must be one of ${TRAINING_WORKLOAD_KINDS.join(', ')}.`);
  }
  const schemaVersion = asPositiveInteger(payload.schemaVersion, `${contextLabel}.schemaVersion`);
  if (schemaVersion !== TRAINING_WORKLOAD_SCHEMA_VERSION) {
    throw new Error(`${contextLabel}.schemaVersion must be ${TRAINING_WORKLOAD_SCHEMA_VERSION}.`);
  }

  const workload = {
    schemaVersion,
    kind,
    id: asNonEmptyString(payload.id, `${contextLabel}.id`),
    description: asNonEmptyString(payload.description, `${contextLabel}.description`),
    claimBoundary: asNonEmptyString(payload.claimBoundary, `${contextLabel}.claimBoundary`),
    seed: asPositiveInteger(payload.seed, `${contextLabel}.seed`),
    baseModelId: asNonEmptyString(payload.baseModelId, `${contextLabel}.baseModelId`),
    studentModelId: asNonEmptyString(payload.studentModelId, `${contextLabel}.studentModelId`, { optional: true }),
    teacherModelId: asNonEmptyString(payload.teacherModelId, `${contextLabel}.teacherModelId`, { optional: true }),
    datasetId: asNonEmptyString(payload.datasetId, `${contextLabel}.datasetId`),
    datasetPath: asNonEmptyString(payload.datasetPath, `${contextLabel}.datasetPath`),
    evalDatasets: normalizeEvalDatasets(payload.evalDatasets ?? [], `${contextLabel}.evalDatasets`),
    trainingSchemaVersion: asPositiveInteger(
      payload.trainingSchemaVersion,
      `${contextLabel}.trainingSchemaVersion`
    ),
    checkpointEvery: asPositiveInteger(payload.checkpointEvery, `${contextLabel}.checkpointEvery`),
    selectionMetric: asNonEmptyString(payload.selectionMetric, `${contextLabel}.selectionMetric`),
    selectionGoal: asEnum(
      payload.selectionGoal,
      `${contextLabel}.selectionGoal`,
      TRAINING_SELECTION_GOALS
    ),
    surfaceSupport: asEnum(
      payload.surfaceSupport,
      `${contextLabel}.surfaceSupport`,
      TRAINING_WORKLOAD_SURFACE_SUPPORT
    ),
    training: normalizeTrainingConfig(payload.training, `${contextLabel}.training`),
    pipeline: null,
  };

  if (kind === 'lora') {
    workload.pipeline = normalizeLoraConfig(payload.lora ?? payload.pipeline, `${contextLabel}.lora`);
  } else if (kind === 'distill') {
    workload.pipeline = normalizeDistillConfig(payload.distill ?? payload.pipeline, `${contextLabel}.distill`);
    const stageRequiresTeacher = workload.pipeline.stagePlan.some((stage) => stage.objective !== 'sft');
    if (stageRequiresTeacher && !workload.teacherModelId) {
      throw new Error(`${contextLabel}.teacherModelId is required when stagePlan includes non-SFT stages.`);
    }
    if (!workload.studentModelId) {
      throw new Error(`${contextLabel}.studentModelId is required for distill workloads.`);
    }
  }

  return withHashes(workload);
}

function withHashes(workload) {
  const configHash = sha256Hex(stableJson(workload));
  return {
    ...workload,
    configHash,
  };
}

async function readRegistryEntryById(registryPath, workloadId) {
  const absoluteRegistryPath = resolve(String(registryPath));
  const raw = await readFile(absoluteRegistryPath, 'utf8');
  const parsed = JSON.parse(raw);
  const workloads = Array.isArray(parsed?.workloads) ? parsed.workloads : [];
  const match = workloads.find((entry) => String(entry?.id || '').trim() === workloadId);
  if (!match) {
    throw new Error(`training workload id "${workloadId}" not found in registry ${absoluteRegistryPath}.`);
  }
  const relativePath = asNonEmptyString(match.path, `${absoluteRegistryPath}.workloads[].path`);
  return resolve(relativePath);
}

export async function loadTrainingWorkloadPack(input, options = {}) {
  const normalizedInput = asNonEmptyString(input, 'workload input');
  const looksLikePath = normalizedInput.endsWith('.json') || normalizedInput.includes('/') || normalizedInput.includes('\\');
  const absolutePath = looksLikePath
    ? resolve(normalizedInput)
    : await readRegistryEntryById(
      options.registryPath || 'tools/configs/training-workloads/registry.json',
      normalizedInput
    );
  const raw = await readFile(absolutePath, 'utf8');
  const parsed = JSON.parse(raw);
  const normalized = normalizeTrainingWorkloadPack(parsed, {
    label: absolutePath,
  });
  return {
    absolutePath,
    path: absolutePath,
    raw,
    workloadSha256: sha256Hex(raw),
    workload: normalized,
  };
}

export function serializeTrainingWorkloadLock(loadedWorkload) {
  return stableJson({
    schemaVersion: TRAINING_WORKLOAD_SCHEMA_VERSION,
    artifactType: 'training_workload_lock',
    workloadId: loadedWorkload.workload.id,
    workloadPath: loadedWorkload.absolutePath,
    workloadSha256: loadedWorkload.workloadSha256,
    workload: loadedWorkload.workload,
  });
}
