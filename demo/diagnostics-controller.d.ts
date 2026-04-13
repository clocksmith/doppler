/**
 * Diagnostics controller for browser demo workflows.
 *
 * @module demo/diagnostics-controller
 */

import type { BrowserSuiteResult } from '../src/inference/browser-harness.js';
import type { RuntimeConfigSchema } from '../src/config/schema/index.js';
import type { SavedReportInfo } from '../src/storage/reports.js';
import type { log as debugLog } from '../src/debug/index.js';
import type { ToolingErrorEnvelope, ToolingSuccessEnvelope } from '../src/tooling/command-envelope.js';
import type { ToolingWorkload } from '../src/tooling/command-api.js';

export interface DiagnosticsControllerOptions {
  log?: typeof debugLog;
  runCommand?: (
    request: Record<string, unknown>
  ) => Promise<ToolingSuccessEnvelope<BrowserSuiteResult> | ToolingErrorEnvelope>;
}

export interface DiagnosticsSuiteOptions {
  workload?: ToolingWorkload;
  suite?: string;
  runtimeConfig?: Partial<RuntimeConfigSchema>;
  runtimeConfigUrl?: string | null;
  runtimeProfile?: string | null;
  captureOutput?: boolean;
  modelId?: string | null;
  modelUrl?: string | null;
  keepPipeline?: boolean;
  report?: Record<string, unknown>;
}

export declare class DiagnosticsController {
  constructor(options?: DiagnosticsControllerOptions);

  log: typeof debugLog;
  runCommand: (
    request: Record<string, unknown>
  ) => Promise<ToolingSuccessEnvelope<BrowserSuiteResult> | ToolingErrorEnvelope>;
  lastReport: Record<string, unknown> | null;
  lastReportInfo: SavedReportInfo | null;

  requireIntent(runtimeConfig: Partial<RuntimeConfigSchema>): string | null;
  applyRuntimeProfile(profileId: string): Promise<Record<string, unknown>>;
  verifySuite(model: Record<string, unknown> | null, options?: DiagnosticsSuiteOptions): Promise<BrowserSuiteResult>;
  runSuite(model: Record<string, unknown> | null, options?: DiagnosticsSuiteOptions): Promise<BrowserSuiteResult>;
}
