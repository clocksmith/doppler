
import { trainStep } from './trainer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { AdamOptimizer } from './optimizer.js';
import { DynamicLossScaler, detectOverflow } from './loss-scaling.js';
import { readBuffer, uploadData } from '../memory/buffer-pool.js';
import { f16ToF32Array } from '../inference/kv-cache/types.js';
import { DataLoader } from './dataloader.js';
import { createCrossEntropyObjective } from './objectives/cross_entropy.js';
import { createDistillKdObjective } from './objectives/distill_kd.js';
import { createDistillTripletObjective } from './objectives/distill_triplet.js';
import { createUlStage1JointObjective } from './objectives/ul_stage1_joint.js';
import { createUlStage2BaseObjective } from './objectives/ul_stage2_base.js';
import {
  createDistillArtifactSession,
  createUlArtifactSession,
  resolveDistillTrainingContract,
  resolveStageAArtifactContext,
  resolveUlTrainingContract,
  resolveStage1ArtifactContext,
} from './artifacts.js';
import { loadCheckpoint, saveCheckpoint } from './checkpoint.js';
import { validateTrainingMetricsEntry } from '../config/schema/training-metrics.schema.js';

function toFloat32(buffer, dtype) {
  if (dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(buffer));
  }
  return new Float32Array(buffer);
}

async function computeLossMean(loss) {
  const data = toFloat32(await readBuffer(loss.buffer), loss.dtype);
  if (!data.length) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) {
    sum += data[i];
  }
  return sum / data.length;
}

async function resolveBatches(dataset, batchSize, shuffle) {
  if (dataset && typeof dataset.batches === 'function') {
    return dataset.batches();
  }
  if (Array.isArray(dataset)) {
    const loader = new DataLoader(dataset, batchSize, shuffle);
    return loader.batches();
  }
  throw new Error('TrainingRunner requires dataset array or DataLoader');
}

function resolveTrainingObjective(config, options) {
  if (options.trainingObjective) {
    return options.trainingObjective;
  }
  const distill = config.training?.distill;
  if (distill?.enabled) {
    if (distill.stage === 'stage_a') {
      return createDistillKdObjective({ crossEntropyLoss: options.crossEntropyLoss });
    }
    if (distill.stage === 'stage_b') {
      return createDistillTripletObjective({ crossEntropyLoss: options.crossEntropyLoss });
    }
  }
  const ul = config.training?.ul;
  if (!ul?.enabled) {
    return createCrossEntropyObjective({ crossEntropyLoss: options.crossEntropyLoss });
  }
  if (ul.stage === 'stage1_joint') {
    return createUlStage1JointObjective({ crossEntropyLoss: options.crossEntropyLoss });
  }
  if (ul.stage === 'stage2_base') {
    return createUlStage2BaseObjective({ crossEntropyLoss: options.crossEntropyLoss });
  }
  return createCrossEntropyObjective({ crossEntropyLoss: options.crossEntropyLoss });
}

function toMetricNumber(value, fallback = null) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return value;
}

function resolveTelemetrySettings(config) {
  const telemetry = config?.training?.telemetry || {};
  const mode = telemetry.mode === 'window' || telemetry.mode === 'epoch'
    ? telemetry.mode
    : 'step';
  const windowSize = Math.max(1, Math.floor(Number(telemetry.windowSize) || 1));
  const emitNaNInfCounters = telemetry.emitNaNInfCounters !== false;
  const alerts = telemetry.alerts && typeof telemetry.alerts === 'object'
    ? telemetry.alerts
    : {};
  const thresholds = alerts.thresholds && typeof alerts.thresholds === 'object'
    ? alerts.thresholds
    : {};
  const normalizeThreshold = (value) => {
    if (value === null || value === undefined) return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  return {
    mode,
    windowSize,
    emitNaNInfCounters,
    alertsEnabled: alerts.enabled === true,
    failOnAlert: alerts.failOnAlert === true,
    thresholds: {
      maxStepTimeMs: normalizeThreshold(thresholds.maxStepTimeMs),
      maxGradientNorm: normalizeThreshold(thresholds.maxGradientNorm),
      maxNaNCount: normalizeThreshold(thresholds.maxNaNCount),
      maxInfCount: normalizeThreshold(thresholds.maxInfCount),
      maxSaturationCount: normalizeThreshold(thresholds.maxSaturationCount),
      minEffectiveLr: normalizeThreshold(thresholds.minEffectiveLr),
    },
  };
}

function pushRolling(windowValues, value, maxSize) {
  windowValues.push(value);
  while (windowValues.length > maxSize) {
    windowValues.shift();
  }
}

function average(values) {
  if (!Array.isArray(values) || values.length === 0) return null;
  let sum = 0;
  for (const value of values) {
    sum += value;
  }
  return sum / values.length;
}

function resolveObjectiveStage(objectiveName) {
  if (objectiveName === 'ul_stage1_joint') return 'stage1_joint';
  if (objectiveName === 'ul_stage2_base') return 'stage2_base';
  return null;
}

function resolveObjectiveDistillStage(objectiveName) {
  if (objectiveName === 'kd') return 'stage_a';
  if (objectiveName === 'triplet') return 'stage_b';
  return null;
}

function toPositiveIntegerOrNull(value) {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  const floored = Math.floor(parsed);
  return floored >= 1 ? floored : null;
}

function resolveProgressContext(config, runOptions = {}) {
  const distillConfig = config?.training?.distill || {};
  const shardIndexInput = toPositiveIntegerOrNull(
    runOptions.distillShardIndex ?? distillConfig.shardIndex ?? null
  );
  const shardCountInput = toPositiveIntegerOrNull(
    runOptions.distillShardCount ?? distillConfig.shardCount ?? null
  );
  if (
    shardIndexInput !== null
    && shardCountInput !== null
    && shardIndexInput > shardCountInput
  ) {
    throw new Error('TrainingRunner: distillShardIndex must be <= distillShardCount.');
  }
  const shardCount = shardCountInput ?? 1;
  const shardIndex = shardIndexInput ?? 1;
  const stepsPerShard = toPositiveIntegerOrNull(runOptions.maxSteps);
  return {
    shardIndex: Math.min(Math.max(1, shardIndex), shardCount),
    shardCount: Math.max(1, shardCount),
    stepsPerShard,
  };
}

function buildProgressSnapshot(step, elapsedMs, context) {
  const shardIndex = context?.shardIndex ?? 1;
  const shardCount = context?.shardCount ?? 1;
  const stepsPerShard = context?.stepsPerShard ?? null;
  const stepInShard = stepsPerShard !== null
    ? Math.min(step, stepsPerShard)
    : step;
  const globalStep = stepsPerShard !== null
    ? (((shardIndex - 1) * stepsPerShard) + stepInShard)
    : null;
  const globalSteps = stepsPerShard !== null
    ? (stepsPerShard * shardCount)
    : null;
  const percentComplete = (
    Number.isFinite(globalStep)
    && Number.isFinite(globalSteps)
    && globalSteps > 0
  )
    ? Math.min(100, (globalStep / globalSteps) * 100)
    : null;
  let etaMs = null;
  if (
    Number.isFinite(globalStep)
    && Number.isFinite(globalSteps)
    && globalStep > 0
    && globalSteps >= globalStep
    && Number.isFinite(elapsedMs)
  ) {
    const meanStepMs = elapsedMs / globalStep;
    const remainingSteps = globalSteps - globalStep;
    if (Number.isFinite(meanStepMs)) {
      etaMs = Math.max(0, remainingSteps * meanStepMs);
    }
  }
  return {
    shardIndex,
    shardCount,
    stepInShard,
    stepsPerShard,
    globalStep,
    globalSteps,
    percentComplete,
    elapsedMs: Number.isFinite(elapsedMs) ? Math.max(0, elapsedMs) : null,
    etaMs,
    etaIso: Number.isFinite(etaMs) ? new Date(Date.now() + etaMs).toISOString() : null,
  };
}

function countNumericAnomaliesFromObject(value, counters) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return;
  for (const candidate of Object.values(value)) {
    if (typeof candidate !== 'number') continue;
    if (Number.isNaN(candidate)) {
      counters.nan += 1;
      continue;
    }
    if (!Number.isFinite(candidate)) {
      counters.inf += 1;
    }
  }
}

function evaluateTelemetryAlerts(entry, telemetry) {
  if (!telemetry?.alertsEnabled) return [];
  const alerts = [];
  const thresholds = telemetry.thresholds || {};
  if (Number.isFinite(thresholds.maxStepTimeMs) && entry.step_time_ms > thresholds.maxStepTimeMs) {
    alerts.push('max_step_time_ms_exceeded');
  }
  if (
    Number.isFinite(thresholds.maxGradientNorm)
    && Number.isFinite(entry.gradient_norm_unclipped)
    && entry.gradient_norm_unclipped > thresholds.maxGradientNorm
  ) {
    alerts.push('max_gradient_norm_exceeded');
  }
  if (Number.isFinite(thresholds.maxNaNCount) && Number.isFinite(entry.nan_count) && entry.nan_count > thresholds.maxNaNCount) {
    alerts.push('max_nan_count_exceeded');
  }
  if (Number.isFinite(thresholds.maxInfCount) && Number.isFinite(entry.inf_count) && entry.inf_count > thresholds.maxInfCount) {
    alerts.push('max_inf_count_exceeded');
  }
  if (
    Number.isFinite(thresholds.maxSaturationCount)
    && Number.isFinite(entry.saturation_count)
    && entry.saturation_count > thresholds.maxSaturationCount
  ) {
    alerts.push('max_saturation_count_exceeded');
  }
  if (
    Number.isFinite(thresholds.minEffectiveLr)
    && Number.isFinite(entry.effective_lr)
    && entry.effective_lr < thresholds.minEffectiveLr
  ) {
    alerts.push('min_effective_lr_below_threshold');
  }
  return alerts;
}

function collectObjectiveMetrics(entry, objectiveMetrics) {
  if (!objectiveMetrics || typeof objectiveMetrics !== 'object') {
    return;
  }
  for (const [key, value] of Object.entries(objectiveMetrics)) {
    if (typeof key !== 'string' || !key) continue;
    if (typeof value === 'number' && Number.isFinite(value)) {
      entry[key] = value;
      continue;
    }
    if (value === null) {
      entry[key] = null;
      continue;
    }
    if (
      (key === 'distill_stage' || key === 'distill_teacher_model_id' || key === 'distill_student_model_id')
      && typeof value === 'string'
      && value.trim()
    ) {
      entry[key] = value.trim();
      continue;
    }
    if (
      (key === 'latent_shape' && Array.isArray(value))
      || ((key === 'latent_clean_values' || key === 'latent_noise_values' || key === 'latent_noisy_values')
        && Array.isArray(value))
    ) {
      entry[key] = value;
    }
  }
}

function resolveModelParamGroups(model) {
  if (model && typeof model.paramGroups === 'function') {
    const groups = model.paramGroups();
    if (!groups || typeof groups !== 'object') {
      throw new Error('model.paramGroups() must return an object of tensor arrays.');
    }
    return groups;
  }
  if (model && typeof model.loraParams === 'function') {
    return { lora: model.loraParams() };
  }
  return {};
}

function selectTrainableParamGroups(paramGroups, freezeMap) {
  const trainableGroups = {};
  const frozenGroups = [];
  for (const [groupName, params] of Object.entries(paramGroups)) {
    const normalizedParams = Array.isArray(params) ? params.filter(Boolean) : [];
    if (freezeMap?.[groupName] === true) {
      frozenGroups.push(groupName);
      continue;
    }
    trainableGroups[groupName] = normalizedParams;
  }
  return { trainableGroups, frozenGroups };
}

function flattenUniqueParams(paramGroups) {
  const unique = new Set();
  const output = [];
  for (const params of Object.values(paramGroups)) {
    for (const tensor of params) {
      if (!tensor || unique.has(tensor)) continue;
      unique.add(tensor);
      output.push(tensor);
    }
  }
  return output;
}

function isTensorLike(value) {
  return !!value
    && typeof value === 'object'
    && Array.isArray(value.shape)
    && value.buffer != null;
}

function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

function normalizeOptionalString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function toBase64(bytes) {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
  }
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function fromBase64(base64) {
  if (typeof Buffer !== 'undefined') {
    const buffer = Buffer.from(base64, 'base64');
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function serializeTensorSnapshot(tensor) {
  const raw = await readBuffer(tensor.buffer);
  const bytes = new Uint8Array(raw);
  return {
    dtype: tensor.dtype,
    shape: Array.isArray(tensor.shape) ? [...tensor.shape] : [],
    dataBase64: toBase64(bytes),
  };
}

async function restoreTensorSnapshot(tensor, snapshot) {
  if (!isTensorLike(tensor)) return false;
  if (!snapshot || typeof snapshot !== 'object') return false;
  const encoded = normalizeOptionalString(snapshot.dataBase64);
  if (!encoded) return false;
  const decoded = fromBase64(encoded);
  uploadData(tensor.buffer, decoded);
  return true;
}

function buildTrainableParamRefs(model, freezeMap = null) {
  const paramGroups = resolveModelParamGroups(model);
  const { trainableGroups } = selectTrainableParamGroups(paramGroups, freezeMap || {});
  const refs = [];
  const seen = new Set();
  for (const [groupName, params] of Object.entries(trainableGroups)) {
    for (let index = 0; index < params.length; index += 1) {
      const tensor = params[index];
      if (!isTensorLike(tensor) || seen.has(tensor)) continue;
      seen.add(tensor);
      refs.push({
        key: `${groupName}[${index}]`,
        tensor,
      });
    }
  }
  return refs;
}

function looksLikeTrainingCheckpointRecord(value) {
  const trainingState = value?.trainingState;
  if (!trainingState || typeof trainingState !== 'object') return false;
  const progress = trainingState.progress;
  if (!progress || typeof progress !== 'object') return false;
  return Number.isInteger(progress.step) && progress.step >= 0;
}

async function createTrainingCheckpointPayload(model, optimizer, context) {
  const freezeMap = context.config?.training?.ul?.freeze
    ?? context.config?.training?.distill?.freeze
    ?? {};
  const refs = buildTrainableParamRefs(model, freezeMap);
  const params = {};
  const optimizerSlots = {};
  for (const ref of refs) {
    params[ref.key] = await serializeTensorSnapshot(ref.tensor);
    const optimizerStateEntry = optimizer?.state instanceof Map
      ? optimizer.state.get(ref.tensor)
      : null;
    if (optimizerStateEntry?.m && optimizerStateEntry?.v) {
      optimizerSlots[ref.key] = {
        m: await serializeTensorSnapshot(optimizerStateEntry.m),
        v: await serializeTensorSnapshot(optimizerStateEntry.v),
      };
    }
  }
  return {
    trainingState: {
      schemaVersion: 1,
      progress: {
        step: context.step,
        epoch: context.epoch,
        batch: context.batch,
      },
      optimizerStepCount: Number.isInteger(optimizer?.stepCount) ? optimizer.stepCount : 0,
      params,
      optimizerSlots,
    },
  };
}

async function restoreTrainingCheckpointState(model, optimizer, checkpointRecord, config) {
  if (!looksLikeTrainingCheckpointRecord(checkpointRecord)) {
    return null;
  }
  const trainingState = checkpointRecord.trainingState;
  const freezeMap = config?.training?.ul?.freeze
    ?? config?.training?.distill?.freeze
    ?? {};
  const refs = buildTrainableParamRefs(model, freezeMap);
  const refMap = new Map(refs.map((entry) => [entry.key, entry.tensor]));
  const params = trainingState.params && typeof trainingState.params === 'object'
    ? trainingState.params
    : {};
  for (const [key, snapshot] of Object.entries(params)) {
    const tensor = refMap.get(key);
    if (!tensor) continue;
    await restoreTensorSnapshot(tensor, snapshot);
  }
  if (optimizer && Number.isInteger(trainingState.optimizerStepCount)) {
    optimizer.stepCount = trainingState.optimizerStepCount;
  }
  const optimizerSlots = trainingState.optimizerSlots && typeof trainingState.optimizerSlots === 'object'
    ? trainingState.optimizerSlots
    : {};
  if (optimizer && typeof optimizer.getState === 'function') {
    for (const [key, snapshot] of Object.entries(optimizerSlots)) {
      const tensor = refMap.get(key);
      if (!tensor) continue;
      const slot = optimizer.getState(tensor);
      if (slot?.m) {
        await restoreTensorSnapshot(slot.m, snapshot?.m);
      }
      if (slot?.v) {
        await restoreTensorSnapshot(slot.v, snapshot?.v);
      }
    }
  }
  const progress = trainingState.progress || {};
  return {
    step: Number.isInteger(progress.step) ? progress.step : 0,
    epoch: Number.isInteger(progress.epoch) ? progress.epoch : 0,
    batch: Number.isInteger(progress.batch) ? progress.batch : 0,
    checkpointHash: checkpointRecord?.metadata?.checkpointHash || null,
  };
}

async function resolveDefaultCheckpointKey(runOptions, distillContract, ulContract) {
  const artifactDir = normalizeOptionalString(
    runOptions.distillArtifactDir
    || runOptions.ulArtifactDir
    || distillContract?.artifactDir
    || ulContract?.artifactDir
  );
  if (!artifactDir) return null;
  if (!isNodeRuntime()) {
    const mode = distillContract?.enabled ? 'distill' : (ulContract?.enabled ? 'ul' : 'training');
    return `${mode}.latest.checkpoint`;
  }
  const { resolve, join } = await import('node:path');
  const mode = distillContract?.enabled ? 'distill' : (ulContract?.enabled ? 'ul' : 'training');
  return resolve(join(artifactDir, `${mode}.latest.checkpoint.json`));
}

async function resolveCheckpointKey(runOptions, distillContract, ulContract) {
  const explicit = normalizeOptionalString(runOptions.resumeFrom);
  if (explicit) {
    return { explicit, fallback: await resolveDefaultCheckpointKey(runOptions, distillContract, ulContract) };
  }
  return { explicit: null, fallback: await resolveDefaultCheckpointKey(runOptions, distillContract, ulContract) };
}

export class TrainingRunner {
  constructor(config, options = {}) {
    this.config = config;
    this.optimizer = options.optimizer || new AdamOptimizer(config);
    this.lossFn = options.crossEntropyLoss || crossEntropyLoss;
    this.clipFn = options.clipGradients || clipGradients;
    this.trainingObjective = resolveTrainingObjective(config, options);
    this.lossScaler = options.lossScaler || new DynamicLossScaler(config.training.lossScaling);
    this.onStep = options.onStep || null;
    this.onEpoch = options.onEpoch || null;
    this.lastArtifact = null;
    this.lastCheckpoint = null;
    this.resumeState = null;
  }

  async run(model, dataset, options = {}) {
    const {
      epochs = 1,
      batchSize = 1,
      shuffle = true,
      maxSteps = null,
      logEvery = 1,
      prepareBatch = null,
    } = options;

    const distillContract = resolveDistillTrainingContract(this.config.training?.distill);
    const ulContract = resolveUlTrainingContract(this.config.training?.ul);
    if (distillContract.enabled && ulContract.enabled) {
      throw new Error('TrainingRunner cannot run distill and ul modes simultaneously.');
    }
    const checkpointInterval = toPositiveIntegerOrNull(options.checkpointEvery) ?? 1;
    const checkpointKeys = await resolveCheckpointKey(options, distillContract, ulContract);
    let checkpointKey = checkpointKeys.fallback;
    let restoredProgress = null;
    const tryRestoreCheckpoint = async (key) => {
      if (!key) return null;
      try {
        const checkpointRecord = await loadCheckpoint(key);
        return restoreTrainingCheckpointState(model, this.optimizer, checkpointRecord, this.config);
      } catch {
        return null;
      }
    };
    if (checkpointKeys.explicit) {
      restoredProgress = await tryRestoreCheckpoint(checkpointKeys.explicit);
      if (restoredProgress) {
        checkpointKey = checkpointKeys.explicit;
      }
    }
    if (!restoredProgress && checkpointKeys.fallback && checkpointKeys.fallback !== checkpointKeys.explicit) {
      restoredProgress = await tryRestoreCheckpoint(checkpointKeys.fallback);
    }
    this.resumeState = restoredProgress || null;
    const persistCheckpoint = async (checkpointContext) => {
      if (!checkpointKey) return;
      const payload = await createTrainingCheckpointPayload(model, this.optimizer, {
        step: checkpointContext.step,
        epoch: checkpointContext.epoch,
        batch: checkpointContext.batch,
        config: this.config,
      });
      await saveCheckpoint(checkpointKey, payload);
      this.lastCheckpoint = {
        key: checkpointKey,
        step: checkpointContext.step,
        epoch: checkpointContext.epoch,
        batch: checkpointContext.batch,
      };
    };

    const artifactSession = distillContract.enabled
      ? await createDistillArtifactSession({
        config: this.config,
        stage: distillContract.stage,
        runOptions: options,
      })
      : (ulContract.enabled
        ? await createUlArtifactSession({
          config: this.config,
          stage: ulContract.stage,
          runOptions: options,
        })
        : null);
    const stage1ArtifactContext = ulContract.enabled && ulContract.stage === 'stage2_base'
      ? await resolveStage1ArtifactContext(this.config)
      : null;
    const stageAArtifactContext = distillContract.enabled && distillContract.stage === 'stage_b'
      ? await resolveStageAArtifactContext(this.config)
      : null;

    let step = restoredProgress?.step || 0;
    let resumeSkipRemaining = restoredProgress?.step || 0;
    const metrics = [];
    const telemetry = resolveTelemetrySettings(this.config);
    const lossWindow = [];
    const stepTimeWindow = [];
    const progressContext = resolveProgressContext(this.config, { ...options, maxSteps });
    const runStartMs = globalThis.performance.now();

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const batches = await resolveBatches(dataset, batchSize, shuffle);
      let batchIndex = 0;
      for await (const rawBatch of batches) {
        if (resumeSkipRemaining > 0) {
          resumeSkipRemaining -= 1;
          continue;
        }
        step += 1;
        batchIndex += 1;
        const batch = prepareBatch ? await prepareBatch(rawBatch) : rawBatch;
        const step_start_ms = globalThis.performance.now();
        const stepResult = await this._runStep(model, batch, {
          stepIndex: step - 1,
          epoch,
          batch: batchIndex,
          stage1ArtifactContext,
          stageAArtifactContext,
        });
        const step_time_ms = globalThis.performance.now() - step_start_ms;
        const progressSnapshot = buildProgressSnapshot(
          step,
          globalThis.performance.now() - runStartMs,
          progressContext
        );
        const meanLoss = await computeLossMean(stepResult.loss);
        pushRolling(lossWindow, meanLoss, telemetry.windowSize);
        pushRolling(stepTimeWindow, step_time_ms, telemetry.windowSize);
        const objectiveName = stepResult.objectiveName || this.trainingObjective?.name || 'cross_entropy';
        const objectiveStage = resolveObjectiveStage(objectiveName);
        const objectiveDistillStage = resolveObjectiveDistillStage(objectiveName);

        const entry = {
          schemaVersion: 1,
          step,
          epoch,
          batch: batchIndex,
          objective: objectiveName,
          total_loss: meanLoss,
          step_time_ms,
          forward_ms: stepResult.forward_ms,
          backward_ms: stepResult.backward_ms,
          optimizer_ms: stepResult.optimizerMetrics?.optimizer_ms,
          effective_lr: toMetricNumber(stepResult.optimizerMetrics?.effective_lr, null),
          scheduler_index: Number.isInteger(stepResult.optimizerMetrics?.scheduler_index)
            ? stepResult.optimizerMetrics.scheduler_index
            : null,
          scheduler_phase: stepResult.optimizerMetrics?.scheduler_phase ?? null,
          gradient_norm_unclipped: stepResult.clipMetrics?.gradient_norm_unclipped,
          gradient_norm_clipped: stepResult.clipMetrics?.gradient_norm_clipped,
          clipped_event_count: stepResult.clipMetrics?.clipped_event_count,
          total_param_count: stepResult.clipMetrics?.total_param_count,
          trainable_param_count: stepResult.paramGroupMetrics?.trainableParamCount ?? null,
          trainable_groups: stepResult.paramGroupMetrics?.trainableGroups ?? [],
          frozen_groups: stepResult.paramGroupMetrics?.frozenGroups ?? [],
          ul_stage: objectiveStage,
          distill_stage: objectiveDistillStage,
          lambda: toMetricNumber(
            stepResult.objectiveMetrics?.lambda,
            objectiveStage ? toMetricNumber(this.config.training?.ul?.lambda0, null) : null
          ),
          progress_shard_index: progressSnapshot.shardIndex,
          progress_shard_count: progressSnapshot.shardCount,
          progress_step_in_shard: progressSnapshot.stepInShard,
          progress_steps_in_shard: progressSnapshot.stepsPerShard,
          progress_global_step: progressSnapshot.globalStep,
          progress_global_steps: progressSnapshot.globalSteps,
          progress_percent_complete: progressSnapshot.percentComplete,
          progress_elapsed_ms: progressSnapshot.elapsedMs,
          progress_eta_ms: progressSnapshot.etaMs,
          progress_eta_iso: progressSnapshot.etaIso,
          telemetry_mode: telemetry.mode,
          telemetry_window_size: telemetry.windowSize,
          window_loss_avg: average(lossWindow),
          window_step_time_ms_avg: average(stepTimeWindow),
        };
        collectObjectiveMetrics(entry, stepResult.objectiveMetrics);
        const anomalies = { nan: 0, inf: 0 };
        if (telemetry.emitNaNInfCounters) {
          countNumericAnomaliesFromObject(entry, anomalies);
          countNumericAnomaliesFromObject(stepResult.objectiveMetrics, anomalies);
        }
        entry.nan_count = anomalies.nan;
        entry.inf_count = anomalies.inf;
        entry.saturation_count = Number.isInteger(stepResult.clipMetrics?.clipped_event_count)
          ? stepResult.clipMetrics.clipped_event_count
          : 0;
        const telemetryAlerts = evaluateTelemetryAlerts(entry, telemetry);
        if (telemetry.alertsEnabled) {
          entry.telemetry_alerts = telemetryAlerts;
        }
        if (telemetry.failOnAlert && telemetryAlerts.length > 0) {
          throw new Error(
            `training telemetry alert(s): ${telemetryAlerts.join(', ')} at step ${entry.step}.`
          );
        }
        validateTrainingMetricsEntry(entry);
        metrics.push(entry);
        if (artifactSession) {
          await artifactSession.appendStep(entry);
        }
        if (
          checkpointKey
          && checkpointInterval !== null
          && (checkpointInterval <= 1 || step % checkpointInterval === 0)
        ) {
          await persistCheckpoint({
            step,
            epoch,
            batch: batchIndex,
          });
        }

        if (this.onStep && (logEvery <= 0 || step % logEvery === 0)) {
          await this.onStep(entry);
        }

        if (maxSteps && step >= maxSteps) {
          if (artifactSession) {
            this.lastArtifact = await artifactSession.finalize(metrics);
          }
          if (this.onEpoch) {
            await this.onEpoch({ epoch, steps: batchIndex, loss: meanLoss });
          }
          if (checkpointKey) {
            await persistCheckpoint({
              step,
              epoch,
              batch: batchIndex,
            });
          }
          return metrics;
        }
      }

      if (this.onEpoch) {
        const last = metrics[metrics.length - 1];
        await this.onEpoch({ epoch, steps: batchIndex, loss: last?.total_loss ?? 0 });
      }
    }

    if (artifactSession) {
      this.lastArtifact = await artifactSession.finalize(metrics);
    } else {
      this.lastArtifact = null;
    }
    if (checkpointKey) {
      const finalEpoch = Math.max(0, epochs - 1);
      const finalBatch = metrics.length > 0
        ? (metrics[metrics.length - 1]?.batch ?? 0)
        : 0;
      await persistCheckpoint({
        step,
        epoch: finalEpoch,
        batch: finalBatch,
      });
    }

    return metrics;
  }

  async _runStep(model, batch, context = {}) {
    const lossScale = this.lossScaler.shouldScale() ? this.lossScaler.scale : 1;
    const options = {
      crossEntropyLoss: this.lossFn,
      clipGradients: this.clipFn,
      optimizer: this.optimizer,
      trainingObjective: this.trainingObjective,
      lossScale,
      stepIndex: context.stepIndex ?? null,
      epochIndex: context.epoch ?? null,
      batchIndex: context.batch ?? null,
      stage1ArtifactContext: context.stage1ArtifactContext ?? null,
      stageAArtifactContext: context.stageAArtifactContext ?? null,
      applyClip: false,
      applyOptimizer: false,
    };

    const result = await trainStep(model, batch, this.config, options);
    let grads = result.grads;

    if (this.lossScaler.enabled && this.lossScaler.overflowCheck) {
      const overflow = await detectOverflow(grads);
      this.lossScaler.update(overflow);
      if (overflow) {
        return {
          loss: result.loss,
          forward_ms: result.forward_ms,
          backward_ms: result.backward_ms,
          clipMetrics: null,
          optimizerMetrics: null,
        };
      }
    } else if (this.lossScaler.enabled) {
      this.lossScaler.update(false);
    }

    const clipMetrics = await this.clipFn(grads, this.config);
    const paramGroups = resolveModelParamGroups(model);
    const freezeMap = this.config.training?.ul?.freeze
      ?? this.config.training?.distill?.freeze
      ?? {};
    const { trainableGroups, frozenGroups } = selectTrainableParamGroups(paramGroups, freezeMap);
    const trainableParams = flattenUniqueParams(trainableGroups);
    const optimizerMetrics = await this.optimizer.step(trainableParams, clipMetrics.clippedGrads, this.config, {
      trainableGroups: Object.keys(trainableGroups),
      frozenGroups,
      allGroups: Object.keys(paramGroups),
    });

    return {
      loss: result.loss,
      forward_ms: result.forward_ms,
      backward_ms: result.backward_ms,
      clipMetrics,
      optimizerMetrics,
      objectiveName: result.objectiveName,
      objectiveMetrics: result.objectiveMetrics,
      paramGroupMetrics: {
        trainableGroups: Object.keys(trainableGroups),
        frozenGroups,
        allGroups: Object.keys(paramGroups),
        trainableParamCount: trainableParams.length,
      },
    };
  }
}

export async function runTraining(model, dataset, config, options = {}) {
  const runner = new TrainingRunner(config, options);
  return runner.run(model, dataset, options);
}
