/**
 * Tooling Intent Config Schema Definitions
 *
 * Defines the top-level tooling intent for verification, investigation,
 * or calibration runs.
 *
 * @module config/schema/tooling
 */

import type { ConverterConfigSchema } from './converter.schema.js';

export type ToolingIntent = 'verify' | 'investigate' | 'calibrate' | null;
export type ToolingDiagnosticsMode = 'off' | 'on_failure' | 'always';

export interface ToolingConfigSchema {
  /** High-level tooling intent for the run */
  intent: ToolingIntent;
  /** Diagnostics policy for verification runs */
  diagnostics: ToolingDiagnosticsMode;
  /** Optional converter config overrides for browser tooling */
  converter?: Partial<ConverterConfigSchema> | null;
}

/** Allowed tooling intents */
export declare const TOOLING_INTENTS: ToolingIntent[];
/** Allowed diagnostics modes */
export declare const TOOLING_DIAGNOSTICS: ToolingDiagnosticsMode[];
/** Default tooling configuration */
export declare const DEFAULT_TOOLING_CONFIG: ToolingConfigSchema;
