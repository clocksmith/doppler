
import { trainStep } from './trainer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { AdamOptimizer } from './optimizer.js';
import { DynamicLossScaler, detectOverflow } from './loss-scaling.js';
import { readBuffer } from '../memory/buffer-pool.js';
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

    let step = 0;
    const metrics = [];
    const telemetry = resolveTelemetrySettings(this.config);
    const lossWindow = [];
    const stepTimeWindow = [];

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const batches = await resolveBatches(dataset, batchSize, shuffle);
      let batchIndex = 0;
      for await (const rawBatch of batches) {
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
