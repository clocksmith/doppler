import type { Tensor } from '../gpu/tensor.js';
import type { UlTrainingConfigSchema } from '../config/schema/ul-training.schema.js';

export interface UlLatentStats {
  mean: number;
  std: number;
}

export interface UlNoisyLatentBatchResult {
  noisyTensor: Tensor;
  cleanStats: UlLatentStats;
  noiseStats: UlLatentStats;
  noisyStats: UlLatentStats;
  alpha: number;
  sigma: number;
  lambda0: number;
  stepIndex: number;
  cleanValues: Float32Array | null;
  noiseValues: Float32Array | null;
  noisyValues: Float32Array | null;
  shape: number[];
}

export declare function resolveUlNoiseScale(lambda0: number): { alpha: number; sigma: number };
export declare function resolveUlScheduledLambda(
  ulConfig: UlTrainingConfigSchema,
  stepIndex?: number
): number;

export declare function buildNoisyLatentsFromInputTensor(
  inputTensor: Tensor,
  ulConfig: UlTrainingConfigSchema,
  options?: { seed?: number; stepIndex?: number; lambda0?: number; includeValues?: boolean }
): Promise<UlNoisyLatentBatchResult>;

export declare function applyUlStage1Batch(
  batch: Record<string, unknown> & { input: Tensor },
  ulConfig: UlTrainingConfigSchema,
  options?: { seed?: number; stepIndex?: number; lambda0?: number; includeValues?: boolean }
): Promise<Record<string, unknown>>;

export declare function cleanupUlPreparedBatch(batch: Record<string, unknown>): void;

export declare function computeLatentBitrateProxy(
  stats: Record<string, unknown> | null | undefined,
  ulConfig: UlTrainingConfigSchema
): number;
