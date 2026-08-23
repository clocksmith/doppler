/**
 * Harness runtime configuration schema.
 *
 * @module config/schema/harness
 */

export type HarnessMode = 'verify' | 'debug' | 'bench' | 'simulation';
export type HarnessWorkload = 'kernels' | 'inference' | 'embedding' | 'training' | 'diffusion' | 'energy';

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

export interface TsirFixtureHarnessConfigSchema {
  dir: string;
  layerFilter: number[] | null;
  prefillOnly: boolean;
  generationStep: number | null;
}

export interface HarnessConfigSchema {
  mode: HarnessMode;
  workload: HarnessWorkload;
  autorun: boolean;
  skipLoad: boolean;
  modelId: string | null;
  referenceTranscript: {
    enabled: boolean;
    captureLogits: boolean;
    captureKvBytes: boolean;
  };
  tsirFixture: TsirFixtureHarnessConfigSchema | null;
  trainingBench: TrainingBenchConfigSchema;
}

export declare const DEFAULT_HARNESS_CONFIG: HarnessConfigSchema;
