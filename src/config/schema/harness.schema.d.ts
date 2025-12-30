/**
 * Harness runtime configuration schema.
 *
 * @module config/schema/harness
 */

export type HarnessMode = 'kernels' | 'inference' | 'bench' | 'training' | 'simulation' | 'energy';

export interface EbmRecordedBenchDimsSchema {
  M: number;
  K: number;
  H: number;
  O: number;
}

export interface EbmRecordedBenchConfigSchema {
  dims: EbmRecordedBenchDimsSchema;
}

export interface TrainingBenchConfigSchema {
  ebmRecorded: EbmRecordedBenchConfigSchema;
}

export interface HarnessConfigSchema {
  mode: HarnessMode;
  autorun: boolean;
  skipLoad: boolean;
  modelId: string | null;
  trainingBench: TrainingBenchConfigSchema;
}

export declare const DEFAULT_HARNESS_CONFIG: HarnessConfigSchema;
