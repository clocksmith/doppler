import { isTensorLike, resolveModelParamGroups, selectTrainableParamGroups } from '../parameter-groups.js';
import { uploadData } from '../../../memory/buffer-pool.js';

export function buildProgressSnapshot(step, elapsedMs, context) {
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

export function evaluateTelemetryAlerts(entry, telemetry) {
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

export function normalizeOptionalString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

export function fromBase64(base64) {
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

export async function restoreTensorSnapshot(tensor, snapshot) {
  if (!isTensorLike(tensor)) return false;
  if (!snapshot || typeof snapshot !== 'object') return false;
  const encoded = normalizeOptionalString(snapshot.dataBase64);
  if (!encoded) return false;
  const decoded = fromBase64(encoded);
  uploadData(tensor.buffer, decoded);
  return true;
}

export function buildTrainableParamRefs(model, freezeMap = null) {
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

export function looksLikeTrainingCheckpointRecord(value) {
  const trainingState = value?.trainingState;
  if (!trainingState || typeof trainingState !== 'object') return false;
  const progress = trainingState.progress;
  if (!progress || typeof progress !== 'object') return false;
  return Number.isInteger(progress.step) && progress.step >= 0;
}

export async function restoreTrainingCheckpointState(model, optimizer, checkpointRecord, config) {
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
  const resumeAudits = Array.isArray(checkpointRecord?.metadata?.resumeAudits)
    ? checkpointRecord.metadata.resumeAudits
        .filter((entry) => entry && typeof entry === 'object')
        .map((entry) => ({ ...entry }))
    : [];
  return {
    step: Number.isInteger(progress.step) ? progress.step : 0,
    epoch: Number.isInteger(progress.epoch) ? progress.epoch : 0,
    batch: Number.isInteger(progress.batch) ? progress.batch : 0,
    checkpointHash: checkpointRecord?.metadata?.checkpointHash || null,
    previousCheckpointHash: checkpointRecord?.metadata?.lineage?.previousCheckpointHash || null,
    checkpointKey: checkpointRecord?.metadata?.lineage?.checkpointKey || null,
    resumeAudits,
    resumeAuditCount: resumeAudits.length,
  };
}

export { isTensorLike, resolveModelParamGroups, selectTrainableParamGroups };
