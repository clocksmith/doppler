/**
 * Tooling Intent Config Schema Definitions
 *
 * Defines runtime-owned tooling observation and converter configuration.
 * Active command intent lives in ToolingCommandRequest.
 *
 * @module config/schema/tooling
 */

import type { ConverterConfigSchema } from './converter.schema.js';

export type ToolingIntent = 'verify' | 'investigate' | 'calibrate' | null;
export type ToolingDiagnosticsMode = 'off' | 'on_failure' | 'always';
export type RefactorReceiptPolicy = 'off' | 'on_failure' | 'always';

export interface ToolingConfigSchema {
  /** Diagnostics policy for verification runs */
  diagnostics: ToolingDiagnosticsMode;
  /** Permanent behavior-receipt capture policy */
  refactorReceipts: RefactorReceiptPolicy;
  /** Optional converter config overrides for browser tooling */
  converter?: Partial<ConverterConfigSchema> | null;
}

/** Allowed command and runtime-profile metadata intents */
export declare const TOOLING_INTENTS: ToolingIntent[];
/** Allowed diagnostics modes */
export declare const TOOLING_DIAGNOSTICS: ToolingDiagnosticsMode[];
/** Allowed behavior-receipt capture policies */
export declare const REFACTOR_RECEIPT_POLICIES: RefactorReceiptPolicy[];
/** Default tooling configuration */
export declare const DEFAULT_TOOLING_CONFIG: ToolingConfigSchema;
