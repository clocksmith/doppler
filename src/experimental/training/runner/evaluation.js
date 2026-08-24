import { log } from '../../../debug/index.js';
import { trainStep } from '../trainer.js';
import { crossEntropyLoss } from '../loss.js';
import { clipGradients } from '../clip.js';
import { AdamOptimizer } from '../optimizer.js';
import { DynamicLossScaler, detectOverflow } from '../loss-scaling.js';
import { readBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { f16ToF32Array } from '../../../inference/kv-cache/types.js';
import { DataLoader } from '../dataloader.js';
import { createCrossEntropyObjective } from '../objectives/cross_entropy.js';
import { createDistillKdObjective } from '../objectives/distill_kd.js';
import { createDistillTripletObjective } from '../objectives/distill_triplet.js';
import { createUlStage1JointObjective } from '../objectives/ul_stage1_joint.js';
import { createUlStage2BaseObjective } from '../objectives/ul_stage2_base.js';
import {
  createDistillArtifactSession,
  createUlArtifactSession,
  resolveDistillTrainingContract,
  resolveStageAArtifactContext,
  resolveUlTrainingContract,
  resolveStage1ArtifactContext,
} from '../artifacts.js';
import { loadCheckpoint, saveCheckpoint } from '../checkpoint.js';
import { validateTrainingMetricsEntry } from '../../../config/schema/training-metrics.schema.js';
import { sha256Hex } from '../../../utils/sha256.js';
import { stableSortObject } from '../../../utils/stable-sort-object.js';
import { buildProgressSnapshot, buildTrainableParamRefs, evaluateTelemetryAlerts, normalizeOptionalString, resolveModelParamGroups, restoreTrainingCheckpointState, selectTrainableParamGroups } from './lifecycle.js';

export function resolveTelemetrySettings(config) {
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
