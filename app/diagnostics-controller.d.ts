/**
 * diagnostics-controller.d.ts - Browser diagnostics controller
 *
 * @module app/diagnostics-controller
 */

import type { ModelInfo } from './model-selector.js';
import type { BrowserSuite, BrowserSuiteResult } from '../src/inference/browser-harness.js';
import type { IntegrityResult } from '../src/storage/shard-manager.js';

export interface DiagnosticsControllerCallbacks {
  onSuiteStart?: (suite: string, model: { modelId: string | null; modelUrl?: string; sourceType?: string }) => void;
  onSuiteComplete?: (result: BrowserSuiteResult) => void;
  onSuiteError?: (error: Error) => void;
  onSuiteFinish?: () => void;
  onVerifyStart?: (modelId: string) => void;
  onVerifyComplete?: (result: IntegrityResult) => void;
  onVerifyError?: (error: Error) => void;
  onVerifyFinish?: () => void;
}

export interface DiagnosticsSuiteOptions {
  suite?: BrowserSuite;
  runtimePreset?: string | null;
  prompt?: string;
  maxTokens?: number;
}

export declare class DiagnosticsController {
  constructor(callbacks?: DiagnosticsControllerCallbacks);

  get isRunning(): boolean;
  get isVerifying(): boolean;

  verifyModel(model: ModelInfo): Promise<IntegrityResult | null>;

  runSuite(model: ModelInfo | null, options?: DiagnosticsSuiteOptions): Promise<BrowserSuiteResult | null>;
}
