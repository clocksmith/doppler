/**
 * Harness runtime configuration schema.
 *
 * @module config/schema/harness
 */

export type HarnessMode = 'kernels' | 'inference' | 'bench' | 'training' | 'simulation';

export interface HarnessConfigSchema {
  mode: HarnessMode;
  autorun: boolean;
  skipLoad: boolean;
  modelId: string | null;
}

export declare const DEFAULT_HARNESS_CONFIG: HarnessConfigSchema;
