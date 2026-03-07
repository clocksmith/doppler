export { AutogradTape, OpType } from './autograd.js';
export { buildAttentionSoftmaxCache } from './attention-backward.js';
export { recordAttentionForward } from './attention-forward.js';
export { LoraAdapter } from './lora.js';
export { AdamOptimizer } from './optimizer.js';
export { trainStep } from './trainer.js';
export { crossEntropyLoss } from './loss.js';
export { clipGradients } from './clip.js';
export { exportLoRAAdapter } from './export.js';
export { DynamicLossScaler, detectOverflow } from './loss-scaling.js';
export { TrainingRunner, runTraining } from './runner.js';
export { runTrainingSuite, runTrainingBenchSuite, trainingHarness } from './suite.js';
export { runTrainingOperatorCommand } from './operator-command.js';
export {
  runLoraPipeline,
  evaluateLoraCheckpoint,
  exportLoraCheckpoint,
  watchLoraCheckpoints,
  compareLoraRun,
  qualityGateLoraRun,
} from './lora-pipeline.js';
export type {
  TrainingObjective,
  TrainingObjectiveContext,
  TrainingObjectiveLossResult,
} from './objectives/base.js';
export {
  createTrainingObjective,
  isTrainingObjective,
  createCrossEntropyObjective,
  CROSS_ENTROPY_OBJECTIVE,
  createDistillKdObjective,
  createDistillTripletObjective,
  createUlStage1JointObjective,
  createUlStage2BaseObjective,
} from './objectives/index.js';
export type {
  UlArtifactSession,
  UlArtifactFinalizeResult,
  DistillArtifactSession,
  DistillArtifactFinalizeResult,
} from './artifacts.js';
export * as distillation from './distillation/index.js';
export {
  createDistillArtifactSession,
  resolveDistillTrainingContract,
  resolveStageAArtifactContext,
  createUlArtifactSession,
  resolveUlTrainingContract,
  resolveStage1ArtifactContext,
} from './artifacts.js';
export {
  resolveUlNoiseScale,
  resolveUlScheduledLambda,
  buildNoisyLatentsFromInputTensor,
  applyUlStage1Batch,
  cleanupUlPreparedBatch,
  computeLatentBitrateProxy,
} from './ul_dataset.js';
export * as datasets from './datasets/index.js';
export { DataLoader } from './dataloader.js';
export { saveCheckpoint, loadCheckpoint } from './checkpoint.js';
