import { canonicalizeLegacyEnumValue } from '../../config/legacy-enum-values.js';
function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

export function resolveUlScheduledLambda(ulConfig, stepIndex = 0) {
  const fallback = Number.isFinite(ulConfig?.lambda0) ? ulConfig.lambda0 : 5;
  const schedule = ulConfig?.noiseSchedule;
  if (!schedule || typeof schedule !== 'object') {
    return fallback;
  }

  const steps = Math.max(1, Math.floor(Number(schedule.steps) || 1));
  const step = Math.max(0, Math.floor(Number(stepIndex) || 0));
  const t = steps <= 1 ? 1 : clamp01(step / (steps - 1));
  const minLogSNR = Number.isFinite(schedule.minLogSNR) ? schedule.minLogSNR : fallback;
  const maxLogSNR = Number.isFinite(schedule.maxLogSNR) ? schedule.maxLogSNR : fallback;
  // Legacy spellings (log_snr_linear, log_snr_cosine) are read throughout 0.x.
  const type = typeof schedule.type === 'string' ? canonicalizeLegacyEnumValue(schedule.type) : 'log-snr-linear';

  if (type === 'log-snr-cosine') {
    const cosineT = 0.5 - 0.5 * Math.cos(Math.PI * t);
    return maxLogSNR + ((minLogSNR - maxLogSNR) * cosineT);
  }

  // Default: linear interpolation from max -> min across schedule steps.
  return maxLogSNR + ((minLogSNR - maxLogSNR) * t);
}
