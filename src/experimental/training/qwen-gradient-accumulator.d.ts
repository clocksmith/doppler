import type { Tensor } from '../../gpu/tensor.js';
import type { TrainingConfigSchema } from '../../config/training-defaults.d.ts';
import type { AdamOptimizer } from './optimizer.js';
import type { OptimizerMetrics } from './trainer.d.ts';

export interface QwenGradientAccumulatorEntry {
  name: string;
  parameter: Tensor;
  gradient: Tensor;
}

export declare class QwenGradientAccumulator {
  constructor(options: { accumSteps: number });
  accumSteps: number;
  microstepCount: number;
  entries: Array<QwenGradientAccumulatorEntry & { elementCount: number }>;
  readonly ready: boolean;
  readonly parameterNames: string[];
  accumulate(entries: QwenGradientAccumulatorEntry[]): Promise<{
    microstepCount: number;
    accumSteps: number;
    ready: boolean;
    parameterCount: number;
  }>;
  step(
    optimizer: AdamOptimizer,
    trainingConfig: TrainingConfigSchema
  ): Promise<OptimizerMetrics>;
  reset(): void;
  dispose(): void;
}
