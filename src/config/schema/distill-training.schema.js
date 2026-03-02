export const DISTILL_STAGE_VALUES = Object.freeze(['stage_a', 'stage_b']);

export const DISTILL_TRAINING_SCHEMA_VERSION = 1;

export const DEFAULT_DISTILL_FREEZE_GROUPS = Object.freeze({
  encoder: false,
  prior: false,
  decoder: false,
  base: false,
  lora: false,
});

export const DEFAULT_DISTILL_TRAINING_CONFIG = Object.freeze({
  schemaVersion: DISTILL_TRAINING_SCHEMA_VERSION,
  enabled: false,
  stage: 'stage_a',
  teacherModelId: null,
  studentModelId: null,
  datasetId: null,
  datasetPath: null,
  languagePair: null,
  shardIndex: null,
  shardCount: null,
  resumeFrom: null,
  artifactDir: 'bench/out/distill',
  stageAArtifact: null,
  stageAArtifactHash: null,
  temperature: 1,
  alphaKd: 1,
  alphaCe: 0,
  allowHintFallback: false,
  tripletMargin: 0.2,
  freeze: DEFAULT_DISTILL_FREEZE_GROUPS,
});

function assertFiniteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Distill config: ${label} must be a finite number.`);
  }
}

function assertBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`Distill config: ${label} must be a boolean.`);
  }
}

function assertNullableString(value, label) {
  if (value === null) return;
  if (typeof value === 'string' && value.trim().length > 0) return;
  throw new Error(`Distill config: ${label} must be a non-empty string or null.`);
}

function assertNullablePositiveInteger(value, label) {
  if (value === null) return;
  if (Number.isInteger(value) && value >= 1) return;
  throw new Error(`Distill config: ${label} must be a positive integer or null.`);
}

export function validateDistillTrainingConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('Distill config: expected an object.');
  }
  assertFiniteNumber(config.schemaVersion, 'schemaVersion');
  assertBoolean(config.enabled, 'enabled');
  if (!DISTILL_STAGE_VALUES.includes(config.stage)) {
    throw new Error(`Distill config: stage must be one of ${DISTILL_STAGE_VALUES.join(', ')}.`);
  }
  assertNullableString(config.teacherModelId, 'teacherModelId');
  assertNullableString(config.studentModelId, 'studentModelId');
  assertNullableString(config.datasetId, 'datasetId');
  assertNullableString(config.datasetPath, 'datasetPath');
  assertNullableString(config.languagePair, 'languagePair');
  assertNullablePositiveInteger(config.shardIndex, 'shardIndex');
  assertNullablePositiveInteger(config.shardCount, 'shardCount');
  assertNullableString(config.resumeFrom, 'resumeFrom');
  assertNullableString(config.artifactDir, 'artifactDir');
  assertNullableString(config.stageAArtifact, 'stageAArtifact');
  assertNullableString(config.stageAArtifactHash, 'stageAArtifactHash');
  assertFiniteNumber(config.temperature, 'temperature');
  assertFiniteNumber(config.alphaKd, 'alphaKd');
  assertFiniteNumber(config.alphaCe, 'alphaCe');
  assertBoolean(config.allowHintFallback, 'allowHintFallback');
  assertFiniteNumber(config.tripletMargin, 'tripletMargin');

  const freeze = config.freeze;
  if (!freeze || typeof freeze !== 'object' || Array.isArray(freeze)) {
    throw new Error('Distill config: freeze must be an object.');
  }
  assertBoolean(freeze.encoder, 'freeze.encoder');
  assertBoolean(freeze.prior, 'freeze.prior');
  assertBoolean(freeze.decoder, 'freeze.decoder');
  assertBoolean(freeze.base, 'freeze.base');
  assertBoolean(freeze.lora, 'freeze.lora');

  if (config.stage === 'stage_b' && !config.stageAArtifact) {
    throw new Error('Distill config: stage_b requires stageAArtifact.');
  }
  if (
    config.shardIndex !== null
    && config.shardCount !== null
    && config.shardIndex > config.shardCount
  ) {
    throw new Error('Distill config: shardIndex must be <= shardCount.');
  }

  return config;
}
