import { createDopplerConfig, DEFAULT_TRAINING_SETTINGS } from './schema/index.js';
import { validateUlTrainingConfig } from './schema/ul-training.schema.js';

function mergeTrainingSettings(base, overrides) {
  if (!overrides) {
    const merged = { ...base };
    validateUlTrainingConfig(merged.ul);
    return merged;
  }

  const merged = {
    enabled: overrides.enabled ?? base.enabled,
    lora: { ...base.lora, ...overrides.lora },
    optimizer: {
      ...base.optimizer,
      ...overrides.optimizer,
      scheduler: { ...base.optimizer.scheduler, ...overrides.optimizer?.scheduler },
    },
    gradient: { ...base.gradient, ...overrides.gradient },
    precision: { ...base.precision, ...overrides.precision },
    attention: { ...base.attention, ...overrides.attention },
    telemetry: {
      ...base.telemetry,
      ...overrides.telemetry,
      alerts: {
        ...base.telemetry.alerts,
        ...overrides.telemetry?.alerts,
        thresholds: {
          ...base.telemetry.alerts.thresholds,
          ...overrides.telemetry?.alerts?.thresholds,
        },
      },
    },
    lossScaling: { ...base.lossScaling, ...overrides.lossScaling },
    ul: {
      ...base.ul,
      ...overrides.ul,
      noiseSchedule: { ...base.ul.noiseSchedule, ...overrides.ul?.noiseSchedule },
      priorAlignment: { ...base.ul.priorAlignment, ...overrides.ul?.priorAlignment },
      decoderSigmoidWeight: { ...base.ul.decoderSigmoidWeight, ...overrides.ul?.decoderSigmoidWeight },
      lossWeights: { ...base.ul.lossWeights, ...overrides.ul?.lossWeights },
      freeze: { ...base.ul.freeze, ...overrides.ul?.freeze },
    },
  };
  validateUlTrainingConfig(merged.ul);
  return merged;
}

export function createTrainingConfig(overrides = {}) {
  const dopplerConfig = createDopplerConfig({
    model: overrides.model,
    runtime: overrides.runtime,
  });

  return {
    ...dopplerConfig,
    training: mergeTrainingSettings(DEFAULT_TRAINING_SETTINGS, overrides.training),
  };
}

export const DEFAULT_TRAINING_CONFIG = createTrainingConfig();

let trainingConfig = DEFAULT_TRAINING_CONFIG;

export function getTrainingConfig() {
  return trainingConfig;
}

export function setTrainingConfig(overrides) {
  trainingConfig = createTrainingConfig(overrides);
  return trainingConfig;
}

export function resetTrainingConfig() {
  trainingConfig = DEFAULT_TRAINING_CONFIG;
  return trainingConfig;
}
