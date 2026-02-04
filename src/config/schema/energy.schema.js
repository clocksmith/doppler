export const DEFAULT_ENERGY_STATE_CONFIG = {
  width: 64,
  height: 64,
  channels: 1,
  dtype: 'f16',
};

export const DEFAULT_ENERGY_INIT_CONFIG = {
  mode: 'normal',
  seed: 1337,
  scale: 1.0,
};

export const DEFAULT_ENERGY_TARGET_CONFIG = {
  mode: 'normal',
  seed: 42,
  scale: 1.0,
};

export const DEFAULT_ENERGY_LOOP_CONFIG = {
  maxSteps: 64,
  minSteps: 4,
  stepSize: 0.1,
  gradientScale: 1.0,
  convergenceThreshold: 1e-4,
};

export const DEFAULT_ENERGY_DIAGNOSTICS_CONFIG = {
  readbackEvery: 4,
  traceEvery: 1,
  historyLimit: 128,
};

export const DEFAULT_ENERGY_QUINTEL_RULES = {
  mirrorX: true,
  mirrorY: true,
  diagonal: false,
  count: true,
  center: false,
};

export const DEFAULT_ENERGY_QUINTEL_WEIGHTS = {
  symmetry: 1.0,
  count: 0.2,
  center: 1.0,
  binarize: 0.01,
};

export const DEFAULT_ENERGY_QUINTEL_CLAMP = {
  min: 0.0,
  max: 1.0,
};

export const DEFAULT_ENERGY_QUINTEL_CONFIG = {
  backend: 'cpu',
  size: 5,
  rules: DEFAULT_ENERGY_QUINTEL_RULES,
  weights: DEFAULT_ENERGY_QUINTEL_WEIGHTS,
  clamp: DEFAULT_ENERGY_QUINTEL_CLAMP,
  countTarget: 12,
  centerTarget: 1,
};

export const DEFAULT_ENERGY_CONFIG = {
  problem: 'l2',
  state: DEFAULT_ENERGY_STATE_CONFIG,
  init: DEFAULT_ENERGY_INIT_CONFIG,
  target: DEFAULT_ENERGY_TARGET_CONFIG,
  loop: DEFAULT_ENERGY_LOOP_CONFIG,
  diagnostics: DEFAULT_ENERGY_DIAGNOSTICS_CONFIG,
  quintel: DEFAULT_ENERGY_QUINTEL_CONFIG,
};
