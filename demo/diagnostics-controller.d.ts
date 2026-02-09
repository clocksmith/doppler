/**
 * Diagnostics controller for browser demo workflows.
 *
 * @module demo/diagnostics-controller
 */

import type { BrowserSuite, BrowserSuiteResult } from '@doppler/core/inference/browser-harness.js';
import type { RuntimeConfigSchema } from '@doppler/core/config/schema/index.js';
import type { SavedReportInfo } from '@doppler/core/storage/reports.js';
import type { log as debugLog } from '@doppler/core/debug/index.js';

export interface DiagnosticsControllerOptions {
  log?: typeof debugLog;
}

export interface DiagnosticsSuiteOptions {
  suite?: BrowserSuite;
  runtimeConfig?: Partial<RuntimeConfigSchema>;
  runtimePreset?: string | null;
  captureOutput?: boolean;
  modelId?: string | null;
  modelUrl?: string | null;
  keepPipeline?: boolean;
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
