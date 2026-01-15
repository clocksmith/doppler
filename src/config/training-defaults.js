import { createDopplerConfig, DEFAULT_TRAINING_SETTINGS } from './schema/index.js';

function mergeTrainingSettings(base, overrides) {
  if (!overrides) {
    return { ...base };
  }

  return {
    enabled: overrides.enabled ?? base.enabled,
    lora: { ...base.lora, ...overrides.lora },
    optimizer: { ...base.optimizer, ...overrides.optimizer },
    gradient: { ...base.gradient, ...overrides.gradient },
    precision: { ...base.precision, ...overrides.precision },
    attention: { ...base.attention, ...overrides.attention },
    lossScaling: { ...base.lossScaling, ...overrides.lossScaling },
  };
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
