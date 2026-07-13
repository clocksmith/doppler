import type { Tensor } from '../../gpu/tensor.js';
import type { AdamOptimizer } from './optimizer.js';
import type { QwenHybridLayer } from './qwen-hybrid-decoder-training-module.js';
import type { QwenGradientAccumulator } from './qwen-gradient-accumulator.js';
import type { TrainingConfigSchema } from '../../config/training-defaults.d.ts';

export interface QwenHybridSftMicrostepInputs {
  tokenIds: Tensor;
  embeddingWeight: Tensor;
  layers: QwenHybridLayer[];
  finalNormWeight: Tensor;
  lmHeadWeight: Tensor;
  targets: Tensor;
}

export interface QwenHybridSftMicrostepOptions {
  numTokens: number;
  hiddenSize: number;
  vocabSize: number;
  activeTokenCount: number;
  rmsEps: number;
  optimizer?: AdamOptimizer;
  trainingConfig: TrainingConfigSchema;
  captureGradients?: boolean;
  applyOptimizer?: boolean;
  gradientAccumulator?: QwenGradientAccumulator | null;
  layerCheckpointInterval?: number | null;
}

export declare function runQwenHybridSftMicrostep(
  inputs: QwenHybridSftMicrostepInputs,
  options: QwenHybridSftMicrostepOptions
): Promise<{
  meanLoss: number;
  activeTokenCount: number;
  parameterNames: string[];
  gradientSnapshots: Record<string, Float32Array> | null;
  optimizerMetrics: Record<string, unknown> | null;
  accumulationMetrics: {
    microstepCount: number;
    accumSteps: number;
    ready: boolean;
    parameterCount: number;
  } | null;
}>;
