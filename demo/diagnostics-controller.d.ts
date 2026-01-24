/**
 * Diagnostics controller for browser demo workflows.
 *
 * @module demo/diagnostics-controller
 */

import type { BrowserSuite, BrowserSuiteResult } from '../src/inference/browser-harness.js';
import type { RuntimeConfigSchema } from '../src/config/schema/index.js';
import type { SavedReportInfo } from '../src/storage/reports.js';
import type { log as debugLog } from '../src/debug/index.js';

export interface DiagnosticsControllerOptions {
  log?: typeof debugLog;
}

export interface DiagnosticsSuiteOptions {
  suite?: BrowserSuite;
  runtimeConfig?: Partial<RuntimeConfigSchema>;
  runtimePreset?: string | null;
  modelId?: string | null;
  modelUrl?: string | null;
  prompt?: string;
  maxTokens?: number;
  report?: Record<string, unknown>;
}

export declare class DiagnosticsController {
  constructor(options?: DiagnosticsControllerOptions);

  log: typeof debugLog;
  lastReport: Record<string, unknown> | null;
  lastReportInfo: SavedReportInfo | null;

  requireIntent(runtimeConfig: Partial<RuntimeConfigSchema>, suite: string): string;
  applyRuntimePreset(presetId: string): Promise<Record<string, unknown>>;
  verifySuite(model: Record<string, unknown> | null, options?: DiagnosticsSuiteOptions): Promise<{ ok: true; suite: string }>;
  runSuite(model: Record<string, unknown> | null, options?: DiagnosticsSuiteOptions): Promise<BrowserSuiteResult>;
}
