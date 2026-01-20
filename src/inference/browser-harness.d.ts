/**
 * browser-harness.ts - Browser diagnostics harness
 *
 * @module inference/browser-harness
 */

import type { InitializeResult, RuntimeOverrides, InferenceHarnessOptions } from './test-harness.js';
import type { SavedReportInfo, SaveReportOptions } from '../storage/reports.js';

export interface BrowserHarnessOptions extends InferenceHarnessOptions {
  modelUrl: string;
  modelId?: string;
  report?: Record<string, unknown> | null;
  buildReport?: (
    result: InitializeResult & { runtime: RuntimeOverrides }
  ) => Promise<Record<string, unknown>> | Record<string, unknown>;
  timestamp?: string | Date;
  searchParams?: URLSearchParams;
}

export interface RuntimeConfigLoadOptions {
  baseUrl?: string;
  signal?: AbortSignal;
}

export interface BrowserHarnessResult extends InitializeResult {
  runtime: RuntimeOverrides;
  report: Record<string, unknown>;
  reportInfo: SavedReportInfo;
}

export declare function initializeBrowserHarness(
  options: BrowserHarnessOptions
): Promise<InitializeResult & { runtime: RuntimeOverrides }>;

export declare function loadRuntimeConfigFromUrl(
  url: string,
  options?: RuntimeConfigLoadOptions
): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;

export declare function applyRuntimeConfigFromUrl(
  url: string,
  options?: RuntimeConfigLoadOptions
): Promise<Record<string, unknown>>;

export declare function loadRuntimePreset(
  presetId: string,
  options?: RuntimeConfigLoadOptions
): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;

export declare function applyRuntimePreset(
  presetId: string,
  options?: RuntimeConfigLoadOptions
): Promise<Record<string, unknown>>;

export declare function saveBrowserReport(
  modelId: string,
  report: Record<string, unknown>,
  options?: SaveReportOptions
): Promise<SavedReportInfo>;

export declare function runBrowserHarness(
  options: BrowserHarnessOptions
): Promise<BrowserHarnessResult>;
