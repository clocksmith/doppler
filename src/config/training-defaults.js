import { createDopplerConfig, DEFAULT_TRAINING_SETTINGS } from './schema/index.js';
import { validateDistillTrainingConfig } from './schema/distill-training.schema.js';
import { validateUlTrainingConfig } from './schema/ul-training.schema.js';

function cloneConfigTree(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function mergeTrainingSettings(base, overrides) {
  const baseConfig = cloneConfigTree(base);
  if (!overrides) {
    const merged = baseConfig;
    validateDistillTrainingConfig(merged.distill);
    validateUlTrainingConfig(merged.ul);
    if (merged.distill.enabled === true && merged.ul.enabled === true) {
      throw new Error('training config: distill and ul modes cannot both be enabled.');
    }
    return merged;
  }

  const merged = {
    enabled: overrides.enabled ?? baseConfig.enabled,
    lora: { ...baseConfig.lora, ...overrides.lora },
    optimizer: {
      ...baseConfig.optimizer,
      ...overrides.optimizer,
      scheduler: { ...baseConfig.optimizer.scheduler, ...overrides.optimizer?.scheduler },
    },
    gradient: { ...baseConfig.gradient, ...overrides.gradient },
    precision: { ...baseConfig.precision, ...overrides.precision },
    attention: { ...baseConfig.attention, ...overrides.attention },
    telemetry: {
      ...baseConfig.telemetry,
      ...overrides.telemetry,
      alerts: {
        ...baseConfig.telemetry.alerts,
        ...overrides.telemetry?.alerts,
        thresholds: {
          ...baseConfig.telemetry.alerts.thresholds,
          ...overrides.telemetry?.alerts?.thresholds,
        },
      },
    },
    lossScaling: { ...baseConfig.lossScaling, ...overrides.lossScaling },
    distill: {
      ...baseConfig.distill,
      ...overrides.distill,
      freeze: { ...baseConfig.distill.freeze, ...overrides.distill?.freeze },
    },
    ul: {
      ...baseConfig.ul,
      ...overrides.ul,
      noiseSchedule: { ...baseConfig.ul.noiseSchedule, ...overrides.ul?.noiseSchedule },
      priorAlignment: { ...baseConfig.ul.priorAlignment, ...overrides.ul?.priorAlignment },
      decoderSigmoidWeight: { ...baseConfig.ul.decoderSigmoidWeight, ...overrides.ul?.decoderSigmoidWeight },
      lossWeights: { ...baseConfig.ul.lossWeights, ...overrides.ul?.lossWeights },
      freeze: { ...baseConfig.ul.freeze, ...overrides.ul?.freeze },
    },
  };
  validateDistillTrainingConfig(merged.distill);
  validateUlTrainingConfig(merged.ul);
  if (merged.distill.enabled === true && merged.ul.enabled === true) {
    throw new Error('training config: distill and ul modes cannot both be enabled.');
  }
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

let trainingConfig = createTrainingConfig();

export function getTrainingConfig() {
  return trainingConfig;
}

export function setTrainingConfig(overrides) {
  trainingConfig = createTrainingConfig(overrides);
  return trainingConfig;
}

export function resetTrainingConfig() {
  trainingConfig = createTrainingConfig();
  return trainingConfig;
}
