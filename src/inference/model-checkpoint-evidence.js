import { sha256Hex } from '../formats/sha256.js';

export const MODEL_CHECKPOINT_EVIDENCE_SCHEMA = 'doppler.model-checkpoint-evidence/v1';

export const MODEL_CHECKPOINT_STAGES = Object.freeze([
  'embedding',
  'rmsnorm',
  'qkv',
  'rope',
  'attention',
  'kv',
  'mlp',
  'logits',
]);

const QKV_STAGES = new Set([
  'attn.qkv_proj',
  'attn.q_proj',
  'attn.k_proj',
  'attn.v_proj',
]);

const ROPE_STAGES = new Set([
  'attn.q_rope',
  'attn.k_rope',
]);

function stableValue(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableValue(entry));
  }
  if (value && typeof value === 'object') {
    const output = {};
    for (const key of Object.keys(value).sort()) {
      if (value[key] !== undefined) {
        output[key] = stableValue(value[key]);
      }
    }
    return output;
  }
  return value;
}

function hashStableJson(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableValue(value)))}`;
}

function isDigest(value) {
  return /^sha256:[0-9a-f]{64}$/u.test(value ?? '');
}

function checkpointStageForRecord(record) {
  const stageName = record?.stageName;
  if (stageName === 'embed.out' || stageName === 'per_layer_embed.out') {
    return 'embedding';
  }
  if (QKV_STAGES.has(stageName)) {
    return 'qkv';
  }
  if (ROPE_STAGES.has(stageName)) {
    return 'rope';
  }
  if (record?.opType === 'normalization') {
    return 'rmsnorm';
  }
  if (record?.opType === 'attention') {
    return 'attention';
  }
  if (record?.opType === 'ffn' || record?.opType === 'activation') {
    return 'mlp';
  }
  if (record?.opType === 'logits') {
    return 'logits';
  }
  return null;
}

function normalizeCheckpointRecord(record, recordIndex) {
  const fullTensorDigest = record?.capture?.fullTensorDigest ?? null;
  if (!isDigest(fullTensorDigest)) {
    return null;
  }
  return {
    recordIndex,
    opId: record.opId ?? null,
    stageName: record.stageName ?? null,
    opType: record.opType ?? null,
    layerIndex: Number.isInteger(record.layerIndex) ? record.layerIndex : null,
    dtype: record.dtype ?? null,
    shapeSignature: record.shapeSignature ?? null,
    fullTensorDigest,
    sample: Array.isArray(record?.capture?.sample)
      ? [...record.capture.sample]
      : null,
    sampleCoordinates: Array.isArray(record?.capture?.sampleCoordinates)
      ? [...record.capture.sampleCoordinates]
      : null,
    data: Array.isArray(record?.capture?.data)
      ? [...record.capture.data]
      : null,
    stats: record?.capture?.stats ?? null,
    hasNaN: record?.capture?.hasNaN === true,
    hasInf: record?.capture?.hasInf === true,
  };
}

function splitTimelineIntoSteps(timeline) {
  const steps = [];
  let current = [];
  for (const record of timeline) {
    if (record?.stageName === 'embed.out' && current.length > 0) {
      steps.push(current);
      current = [];
    }
    current.push(record);
  }
  if (current.length > 0) {
    steps.push(current);
  }
  return steps;
}

function buildStepCheckpoint(stepRecords, stepIndex) {
  const phase = stepIndex === 0 ? 'prefill' : 'decode';
  const recordsByStage = Object.fromEntries(
    MODEL_CHECKPOINT_STAGES
      .filter((stage) => stage !== 'kv')
      .map((stage) => [stage, []])
  );
  for (let recordIndex = 0; recordIndex < stepRecords.length; recordIndex += 1) {
    const source = stepRecords[recordIndex];
    const checkpointStage = checkpointStageForRecord(source);
    if (!checkpointStage || checkpointStage === 'kv') continue;
    const record = normalizeCheckpointRecord(source, recordIndex);
    if (record) {
      recordsByStage[checkpointStage].push(record);
    }
  }

  const checkpoints = {};
  const missing = [];
  for (const [stage, records] of Object.entries(recordsByStage)) {
    if (records.length === 0) {
      missing.push(stage);
      continue;
    }
    checkpoints[stage] = {
      stage,
      phase,
      stepIndex,
      digest: hashStableJson(records.map((record) => ({
        opId: record.opId,
        layerIndex: record.layerIndex,
        dtype: record.dtype,
        shapeSignature: record.shapeSignature,
        fullTensorDigest: record.fullTensorDigest,
      }))),
      recordCount: records.length,
      records,
    };
  }
  return {
    stepIndex,
    phase,
    checkpoints,
    missing,
    pass: missing.length === 0,
  };
}

function normalizeKvCheckpoint(kvCacheByteProof, stepIndex) {
  const layers = Array.isArray(kvCacheByteProof?.layers)
    ? kvCacheByteProof.layers
    : Array.isArray(kvCacheByteProof?.byteDigests)
      ? kvCacheByteProof.byteDigests
      : [];
  const records = layers
    .filter((layer) => isDigest(layer?.keyDigest) && isDigest(layer?.valueDigest))
    .map((layer) => ({
      layerIndex: Number.isInteger(layer.layer) ? layer.layer : null,
      seqLen: Number.isInteger(layer.seqLen) ? layer.seqLen : null,
      keyBytes: Number.isInteger(layer.keyBytes) ? layer.keyBytes : null,
      valueBytes: Number.isInteger(layer.valueBytes) ? layer.valueBytes : null,
      keyDigest: layer.keyDigest,
      valueDigest: layer.valueDigest,
    }));
  if (records.length === 0) return null;
  const declaredDigest = kvCacheByteProof?.digest ?? kvCacheByteProof?.byteDigest ?? null;
  return {
    stage: 'kv',
    phase: stepIndex === 0 ? 'prefill' : 'decode',
    stepIndex,
    digest: isDigest(declaredDigest) ? declaredDigest : hashStableJson(records),
    recordCount: records.length,
    layout: kvCacheByteProof?.layout ?? null,
    kvDtype: kvCacheByteProof?.kvDtype ?? null,
    records,
  };
}

export function buildModelCheckpointEvidence(options) {
  const timeline = options?.operatorDiagnostics?.timeline;
  if (!Array.isArray(timeline) || timeline.length === 0) {
    return null;
  }
  const stepRecords = splitTimelineIntoSteps(timeline);
  const steps = stepRecords.map((records, stepIndex) => (
    buildStepCheckpoint(records, stepIndex)
  ));
  const kv = normalizeKvCheckpoint(options?.kvCacheByteProof, steps.length - 1);
  const expectedStepCount = Number.isInteger(options?.expectedStepCount)
    ? options.expectedStepCount
    : null;
  const minimumDecodeSteps = Number.isInteger(options?.minimumDecodeSteps)
    ? options.minimumDecodeSteps
    : null;
  const decodeStepCount = Math.max(0, steps.length - 1);
  const capturedStages = new Set();
  for (const step of steps) {
    for (const stage of Object.keys(step.checkpoints)) {
      capturedStages.add(stage);
    }
  }
  if (kv) capturedStages.add('kv');
  const missingStages = MODEL_CHECKPOINT_STAGES.filter((stage) => !capturedStages.has(stage));
  const blockers = [];
  if (steps.some((step) => !step.pass)) {
    blockers.push('one or more prefill/decode steps are missing tensor checkpoint classes');
  }
  if (!kv) {
    blockers.push('KV byte digests were not captured');
  }
  if (expectedStepCount !== null && steps.length !== expectedStepCount) {
    blockers.push(`expected ${expectedStepCount} checkpoint steps, captured ${steps.length}`);
  }
  if (minimumDecodeSteps !== null && decodeStepCount < minimumDecodeSteps) {
    blockers.push(`expected at least ${minimumDecodeSteps} decode steps, captured ${decodeStepCount}`);
  }

  return {
    schema: MODEL_CHECKPOINT_EVIDENCE_SCHEMA,
    status: blockers.length === 0 && missingStages.length === 0 ? 'complete' : 'blocked',
    timelineRecordCount: timeline.length,
    expectedStepCount,
    stepCount: steps.length,
    decodeStepCount,
    requiredStages: [...MODEL_CHECKPOINT_STAGES],
    capturedStages: [...capturedStages].sort(),
    missingStages,
    blockers,
    steps,
    kv,
  };
}

export function flattenModelCheckpointDigests(evidence) {
  if (!evidence || !Array.isArray(evidence.steps)) return [];
  const checkpoints = [];
  for (const step of evidence.steps) {
    checkpoints.push(...Object.values(step.checkpoints ?? {}));
  }
  if (evidence.kv) checkpoints.push(evidence.kv);
  return checkpoints;
}
