import { createTrainingConfig } from '../../../config/training-defaults.js';
import { sha256Hex } from '../../../utils/sha256.js';
import { stableSortObject } from '../../../utils/stable-sort-object.js';

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function normalizeStageLabel(value) {
  const normalized = String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  return normalized;
}

export function resolveInternalDistillStage(stageEntry) {
  const trainingStage = normalizeStageLabel(stageEntry?.trainingStage || stageEntry?.id || '');
  const objective = normalizeStageLabel(stageEntry?.objective || '');
  if (trainingStage === 'sft' || objective === 'sft') {
    throw new Error(
      'Distillation workload stage uses "sft", but the current JS distill runner only supports the KD-oriented stage_a contract. Use objective="kd" / trainingStage="stage_a" explicitly.'
    );
  }
  if (
    trainingStage === 'stage_b'
    || trainingStage === 'post_sft_distill'
    || trainingStage === 'post_sft_triplet'
    || objective === 'triplet'
  ) {
    return 'stage_b';
  }
  if (
    trainingStage === 'stage_a'
    || trainingStage === 'kd'
    || objective === 'kd'
    || objective === 'cross_entropy'
  ) {
    return 'stage_a';
  }
  throw new Error(
    `Unsupported distillation stage "${stageEntry?.trainingStage || stageEntry?.id || 'unknown'}".`
  );
}

export function buildDistillationTrainingConfigFromWorkload(loadedWorkload, stageEntry, options = {}) {
  const workload = loadedWorkload.workload;
  if (workload.kind !== 'distill') {
    throw new Error('buildDistillationTrainingConfigFromWorkload requires a distill workload.');
  }
  const internalStage = resolveInternalDistillStage(stageEntry);
  const distillTraining = {
    enabled: true,
    stage: internalStage,
    teacherModelId: workload.teacherModelId,
    studentModelId: workload.studentModelId,
    datasetId: workload.datasetId,
    datasetPath: options.datasetPath || workload.datasetPath,
    sourceLangs: workload.pipeline.sourceLangs,
    targetLangs: workload.pipeline.targetLangs,
    pairAllowlist: workload.pipeline.pairAllowlist,
    strictPairContract: workload.pipeline.strictPairContract === true,
    stageAArtifact: options.stageAArtifact || null,
    stageAArtifactHash: options.stageAArtifactHash || null,
    artifactDir: options.artifactDir || null,
    temperature: workload.pipeline.temperature,
    alphaKd: workload.pipeline.alphaKd,
    alphaCe: workload.pipeline.alphaCe,
    tripletMargin: workload.pipeline.tripletMargin,
    studentGraphMode: workload.pipeline.studentGraphMode,
  };
  if (internalStage === 'stage_b') {
    distillTraining.freeze = {
      encoder: true,
      prior: true,
      decoder: true,
      base: false,
      lora: false,
    };
  }
  const trainingConfig = createTrainingConfig({
    training: {
      enabled: true,
      optimizer: {
        type: workload.training.optimizer.type,
        lr: workload.training.optimizer.lr,
        beta1: workload.training.optimizer.beta1,
        beta2: workload.training.optimizer.beta2,
        eps: workload.training.optimizer.eps,
        weightDecay: workload.training.optimizer.weightDecay,
        scheduler: workload.training.optimizer.scheduler,
      },
      gradient: {
        maxNorm: workload.training.gradientClipping.maxNorm,
      },
      precision: workload.training.precision,
      distill: distillTraining,
    },
  });
  return {
    internalStage,
    trainingConfig,
    trainingConfigHash: sha256Hex(stableJson({
      workloadConfigHash: workload.configHash,
      stageEntry,
      datasetPath: options.datasetPath || workload.datasetPath,
      stageAArtifact: options.stageAArtifact || null,
      stageAArtifactHash: options.stageAArtifactHash || null,
    })),
  };
}
