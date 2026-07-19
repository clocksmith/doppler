import type {
  CausalLmLoraTrainer,
  CausalLmLoraTrainerResult,
} from './experimental/training/lora-pipeline.js';
import type {
  LoadedTrainingWorkload,
  TrainingWorkloadPack,
} from './experimental/training/workloads.js';

export { AutogradTape, OpType } from './experimental/training/autograd.js';
export { buildAttentionSoftmaxCache } from './experimental/training/attention-backward.js';
export { recordAttentionForward } from './experimental/training/attention-forward.js';
export { LoraAdapter } from './experimental/training/lora.js';
export { AdamOptimizer } from './experimental/training/optimizer.js';
export { trainStep } from './experimental/training/trainer.js';
export { crossEntropyLoss } from './experimental/training/loss.js';
export { clipGradients } from './experimental/training/clip.js';
export { exportLoRAAdapter, serializeLoRASafetensors } from './experimental/training/export.js';
export { DynamicLossScaler, detectOverflow } from './experimental/training/loss-scaling.js';
export { TrainingRunner, runTraining } from './experimental/training/runner.js';
export { DataLoader } from './experimental/training/dataloader.js';
export { saveCheckpoint, loadCheckpoint } from './experimental/training/checkpoint.js';
export {
  LORA_RUNNER_BASE_MODEL_REGISTRY,
  LORA_RUNNER_DATASET_FORMAT_REGISTRY,
  LORA_RUNNER_SUPPORT_CONTRACT,
  compareLoraRun,
  evaluateLoraCheckpoint,
  exportLoraCheckpoint,
  getLoraRunnerCompatibility,
  qualityGateLoraRun,
  watchLoraCheckpoints,
} from './experimental/training/lora-pipeline.js';
export {
  loadTrainingWorkloadPack,
  normalizeTrainingWorkloadPack,
  serializeTrainingWorkloadLock,
} from './experimental/training/workloads.js';
export type {
  CausalLmLoraTrainer,
  CausalLmLoraTrainerInput,
  CausalLmLoraTrainerResult,
  CausalLmLoraTrainerTensor,
} from './experimental/training/lora-pipeline.js';
export type {
  LoadedTrainingWorkload,
  LoRAWorkloadPipelineConfig,
  TrainingWorkloadPack,
} from './experimental/training/workloads.js';

export type TrainingBackend = 'webgpu_native' | 'external';

export interface TrainingBackendCapability {
  id: TrainingBackend;
  supported: boolean;
  blockedReasons: readonly string[];
}

export interface TrainingCapabilities {
  schemaVersion: 1;
  scope: 'completion_masked_sft_lora';
  supported: boolean;
  operatorSurface: 'node';
  runnerKey: string;
  baseModelId: string;
  baseModelFamily: string | null;
  datasetFormat: string;
  taskType: string;
  backends: Readonly<{
    webgpuNative: TrainingBackendCapability;
    external: TrainingBackendCapability;
  }>;
  adapterExport: Readonly<{
    format: 'safetensors';
    manifest: 'rdrr_lora_adapter';
    runtimeLoadable: true;
  }>;
  compatibility: ReturnType<typeof import('./experimental/training/lora-pipeline.js').getLoraRunnerCompatibility>;
  blockedReasons: readonly string[];
}

export interface TrainSftLoRAOptions {
  backend: TrainingBackend;
  loadedWorkload: LoadedTrainingWorkload;
  runRoot?: string | null;
  timestamp?: string | Date | null;
  causalLmTrainer?: CausalLmLoraTrainer;
  fetch?: (url: string) => Promise<string>;
  readFile?: (path: string) => Promise<string>;
}

export declare const TRAINING_BACKENDS: readonly TrainingBackend[];

export declare function getTrainingCapabilities(
  workload: TrainingWorkloadPack | Record<string, unknown>
): TrainingCapabilities;

export declare function assertTrainingBackend(
  workload: TrainingWorkloadPack | Record<string, unknown>,
  backend: TrainingBackend
): TrainingCapabilities;

export declare function trainSftLoRA(
  options: TrainSftLoRAOptions
): Promise<CausalLmLoraTrainerResult | Record<string, unknown>>;
