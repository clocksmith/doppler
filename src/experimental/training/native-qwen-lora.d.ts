import type { InferencePipeline } from '../../inference/pipelines/text.js';
import type { LoRAExportResult } from './export.js';
import type { Tensor } from '../../gpu/tensor.js';

export interface NativeQwenLoRATrainerOptions {
  pipeline: InferencePipeline;
  baseModelId: string;
  layerIdx: number;
  module: 'down_proj';
  rank: number;
  alpha: number;
  optimizer: {
    type?: string;
    lr: number;
    beta1?: number;
    beta2?: number;
    eps?: number;
    weightDecay?: number;
  };
  gradient?: { maxNorm?: number };
  precision?: { activations?: 'f16' | 'f32'; gradients?: 'f32'; loraParams?: 'f16' | 'f32' };
}

export interface NativeQwenLoRAStepResult {
  loss: number;
  gradientNorm: number | null;
  optimizer: Record<string, unknown> | null;
}

export interface NativeQwenLoRACheckpointTensor {
  dtype: 'f16' | 'f32';
  shape: number[];
  bytes: string;
}

export interface NativeQwenLoRACheckpoint {
  schemaVersion: 1;
  backend: 'webgpu_native';
  baseModelId: string;
  target: { layerIdx: number; module: 'down_proj' };
  rank: number;
  alpha: number;
  stepCount: number;
  adapter: { A: NativeQwenLoRACheckpointTensor; B: NativeQwenLoRACheckpointTensor };
  optimizer: Array<{ m: NativeQwenLoRACheckpointTensor; v: NativeQwenLoRACheckpointTensor }>;
}

export declare class NativeQwenLoRATrainer {
  constructor(options: NativeQwenLoRATrainerOptions);
  readonly pipeline: InferencePipeline;
  readonly layerIdx: number;
  readonly module: 'down_proj';
  readonly baseModelId: string;
  readonly adapter: { A: Tensor; B: Tensor; rank: number; alpha: number };
  paramGroups(): { base: Tensor[]; lora: Tensor[] };
  trainStep(
    inputIds: readonly number[],
    targetIds: readonly number[],
    supervisedTokenCount: number
  ): Promise<NativeQwenLoRAStepResult>;
  exportAdapter(options: {
    id: string;
    name: string;
    weightsPath: string;
  }): Promise<LoRAExportResult>;
  createCheckpoint(): Promise<NativeQwenLoRACheckpoint>;
  restoreCheckpoint(checkpoint: NativeQwenLoRACheckpoint): void;
  dispose(): void;
}

export declare function createNativeQwenLoRATrainer(
  options: NativeQwenLoRATrainerOptions
): NativeQwenLoRATrainer;

export declare function loadNativeQwenTrainingPipeline(
  modelUrl: string,
  options?: {
    log?: (message: string) => void;
    onProgress?: (stage: string, progress: number, message: string) => void;
    runtime?: Record<string, unknown>;
  }
): Promise<{
  pipeline: InferencePipeline;
  manifest: Record<string, unknown> & { modelId: 'qwen-3-5-0-8b-q4k-ehaf16' };
  capabilities: Record<string, unknown>;
  configSnapshot: Record<string, unknown>;
}>;

export interface NativeQwenSftLoRAOptions extends NativeQwenLoRATrainerOptions {
  samples: Array<{
    inputIds: readonly number[];
    targetIds: readonly number[];
    supervisedTokenCount: number;
  }>;
  maxSteps?: number;
  export?: { id: string; name: string; weightsPath: string } | null;
}

export declare function trainNativeQwenSftLoRA(options: NativeQwenSftLoRAOptions): Promise<{
  backend: 'webgpu_native';
  surfaces: ['browser', 'node', 'bun'];
  baseModelId: string;
  target: { layerIdx: number; module: 'down_proj' };
  metrics: NativeQwenLoRAStepResult[];
  adapter: LoRAExportResult | null;
}>;
