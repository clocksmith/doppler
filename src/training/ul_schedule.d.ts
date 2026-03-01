import type { UlTrainingConfigSchema } from '../config/schema/ul-training.schema.js';

export declare function resolveUlScheduledLambda(
  ulConfig: UlTrainingConfigSchema,
  stepIndex?: number
): number;
