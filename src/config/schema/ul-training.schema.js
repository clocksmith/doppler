export const UL_STAGE_VALUES = Object.freeze(['stage1_joint', 'stage2_base']);
export const UL_TRAINING_SCHEMA_VERSION = 1;

export const DEFAULT_UL_NOISE_SCHEDULE = Object.freeze({
  type: 'log_snr_linear',
  minLogSNR: -4,
  maxLogSNR: 5,
  steps: 64,
});

export const DEFAULT_UL_PRIOR_ALIGNMENT = Object.freeze({
  enabled: true,
  weight: 1,
});

export const DEFAULT_UL_DECODER_SIGMOID_WEIGHT = Object.freeze({
  enabled: true,
  slope: 1,
  midpoint: 0,
});

export const DEFAULT_UL_FREEZE_GROUPS = Object.freeze({
  encoder: false,
  prior: false,
  decoder: false,
  base: false,
  lora: false,
});

export const DEFAULT_UL_LOSS_WEIGHTS = Object.freeze({
  ce: 1,
  prior: 1,
  decoder: 1,
  recon: 1,
});

export const DEFAULT_UL_TRAINING_CONFIG = Object.freeze({
  schemaVersion: UL_TRAINING_SCHEMA_VERSION,
  enabled: false,
  stage: 'stage1_joint',
  lambda0: 5,
  seed: 1337,
  artifactDir: 'bench/out/ul',
  stage1Artifact: null,
  stage1ArtifactHash: null,
  noiseSchedule: DEFAULT_UL_NOISE_SCHEDULE,
  priorAlignment: DEFAULT_UL_PRIOR_ALIGNMENT,
  decoderSigmoidWeight: DEFAULT_UL_DECODER_SIGMOID_WEIGHT,
  lossWeights: DEFAULT_UL_LOSS_WEIGHTS,
  freeze: DEFAULT_UL_FREEZE_GROUPS,
});

function assertFiniteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`UL config: ${label} must be a finite number.`);
  }
}

function assertBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`UL config: ${label} must be a boolean.`);
  }
}

function assertNullableString(value, label) {
  if (value === null) return;
  if (typeof value === 'string' && value.trim().length > 0) return;
  throw new Error(`UL config: ${label} must be a non-empty string or null.`);
}

export function validateUlTrainingConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('UL config: expected an object.');
  }
  assertFiniteNumber(config.schemaVersion, 'schemaVersion');
  if (!UL_STAGE_VALUES.includes(config.stage)) {
    throw new Error(`UL config: stage must be one of ${UL_STAGE_VALUES.join(', ')}.`);
  }
  assertBoolean(config.enabled, 'enabled');
  assertFiniteNumber(config.lambda0, 'lambda0');
  assertFiniteNumber(config.seed, 'seed');
  assertNullableString(config.artifactDir, 'artifactDir');
  assertNullableString(config.stage1Artifact, 'stage1Artifact');
  assertNullableString(config.stage1ArtifactHash, 'stage1ArtifactHash');

  const noiseSchedule = config.noiseSchedule;
  if (!noiseSchedule || typeof noiseSchedule !== 'object' || Array.isArray(noiseSchedule)) {
    throw new Error('UL config: noiseSchedule must be an object.');
  }
  if (typeof noiseSchedule.type !== 'string' || !noiseSchedule.type.trim()) {
    throw new Error('UL config: noiseSchedule.type must be a non-empty string.');
  }
  assertFiniteNumber(noiseSchedule.minLogSNR, 'noiseSchedule.minLogSNR');
  assertFiniteNumber(noiseSchedule.maxLogSNR, 'noiseSchedule.maxLogSNR');
  assertFiniteNumber(noiseSchedule.steps, 'noiseSchedule.steps');

  const priorAlignment = config.priorAlignment;
  if (!priorAlignment || typeof priorAlignment !== 'object' || Array.isArray(priorAlignment)) {
    throw new Error('UL config: priorAlignment must be an object.');
  }
  assertBoolean(priorAlignment.enabled, 'priorAlignment.enabled');
  assertFiniteNumber(priorAlignment.weight, 'priorAlignment.weight');

  const decoderSigmoidWeight = config.decoderSigmoidWeight;
  if (
    !decoderSigmoidWeight
    || typeof decoderSigmoidWeight !== 'object'
    || Array.isArray(decoderSigmoidWeight)
  ) {
    throw new Error('UL config: decoderSigmoidWeight must be an object.');
  }
  assertBoolean(decoderSigmoidWeight.enabled, 'decoderSigmoidWeight.enabled');
  assertFiniteNumber(decoderSigmoidWeight.slope, 'decoderSigmoidWeight.slope');
  assertFiniteNumber(decoderSigmoidWeight.midpoint, 'decoderSigmoidWeight.midpoint');

  const lossWeights = config.lossWeights;
  if (!lossWeights || typeof lossWeights !== 'object' || Array.isArray(lossWeights)) {
    throw new Error('UL config: lossWeights must be an object.');
  }
  assertFiniteNumber(lossWeights.ce, 'lossWeights.ce');
  assertFiniteNumber(lossWeights.prior, 'lossWeights.prior');
  assertFiniteNumber(lossWeights.decoder, 'lossWeights.decoder');
  assertFiniteNumber(lossWeights.recon, 'lossWeights.recon');

  const freeze = config.freeze;
  if (!freeze || typeof freeze !== 'object' || Array.isArray(freeze)) {
    throw new Error('UL config: freeze must be an object.');
  }
  assertBoolean(freeze.encoder, 'freeze.encoder');
  assertBoolean(freeze.prior, 'freeze.prior');
  assertBoolean(freeze.decoder, 'freeze.decoder');
  assertBoolean(freeze.base, 'freeze.base');
  assertBoolean(freeze.lora, 'freeze.lora');

  if (config.stage === 'stage2_base' && !config.stage1Artifact) {
    throw new Error('UL config: stage2_base requires stage1Artifact.');
  }

  return config;
}
