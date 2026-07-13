import type { Tensor } from '../../gpu/tensor.js';
import type { AdamOptimizer } from './optimizer.js';

export interface QwenAdapterTrainingStateEntry {
  name: string;
  parameter: Tensor;
}

export interface QwenAdapterTrainingProgress {
  microstepCount: number;
  optimizerStepCount?: number;
  consumedRowIds: string[];
  consumedPrefixSha256?: string;
}

export interface QwenAdapterTrainingState {
  artifactType: 'qwen_adapter_training_state';
  schemaVersion: 1;
  parameterNames: string[];
  progress: Required<QwenAdapterTrainingProgress>;
  tensors: Record<string, {
    dtype: 'f32';
    shape: number[];
    parameter: { dataBase64: string; dataSha256: string };
    moment1: { dataBase64: string; dataSha256: string };
    moment2: { dataBase64: string; dataSha256: string };
  }>;
  payloadSha256: string;
}

export declare function captureQwenAdapterTrainingState(
  entries: QwenAdapterTrainingStateEntry[],
  optimizer: AdamOptimizer,
  progress: QwenAdapterTrainingProgress
): Promise<QwenAdapterTrainingState>;

export declare function validateQwenAdapterTrainingState(
  payload: QwenAdapterTrainingState
): QwenAdapterTrainingState;

export declare function restoreQwenAdapterTrainingState(
  entries: QwenAdapterTrainingStateEntry[],
  optimizer: AdamOptimizer,
  payload: QwenAdapterTrainingState
): Promise<Required<QwenAdapterTrainingProgress> & {
  parameterCount: number;
  payloadSha256: string;
}>;
