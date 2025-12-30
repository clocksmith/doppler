import type { DopplerConfigSchema } from './schema/doppler.schema.js';
import type { TrainingSettingsSchema } from './schema/training.schema.js';

export interface TrainingConfigSchema extends DopplerConfigSchema {
  training: TrainingSettingsSchema;
}

export interface TrainingConfigOverrides {
  model?: DopplerConfigSchema['model'];
  runtime?: Partial<DopplerConfigSchema['runtime']>;
  training?: Partial<TrainingSettingsSchema>;
}

export declare const DEFAULT_TRAINING_CONFIG: TrainingConfigSchema;

export declare function createTrainingConfig(
  overrides?: TrainingConfigOverrides
): TrainingConfigSchema;

export declare function getTrainingConfig(): TrainingConfigSchema;
export declare function setTrainingConfig(
  overrides?: TrainingConfigOverrides
): TrainingConfigSchema;
export declare function resetTrainingConfig(): TrainingConfigSchema;
