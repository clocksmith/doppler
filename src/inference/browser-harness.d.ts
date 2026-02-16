/**
 * browser-harness.ts - Browser diagnostics harness
 *
 * @module inference/browser-harness
 */

import type { InitializeResult, RuntimeOverrides, InferenceHarnessOptions } from './test-harness.js';
import type { InferencePipeline } from './pipelines/text.js';
import type { DiffusionPipeline } from './pipelines/diffusion/pipeline.js';
import type { EnergyPipeline } from './pipelines/energy/pipeline.js';
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
  presetBaseUrl?: string;
  signal?: AbortSignal;
}

export type BrowserSuite = 'kernels' | 'inference' | 'bench' | 'debug' | 'diffusion' | 'energy';

export interface SuiteTestResult {
  name: string;
  passed: boolean;
  duration: number;
  error?: string;
  skipped?: boolean;
}

export interface SuiteSummary {
  suite: string;
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  results: SuiteTestResult[];
}

export interface DiffusionOutput {
  pixels: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface BrowserSuiteOptions extends InferenceHarnessOptions {
  suite?: BrowserSuite;
  modelUrl?: string;
  modelId?: string;
  runtimePreset?: string | null;
  captureOutput?: boolean;
  keepPipeline?: boolean;
  report?: Record<string, unknown>;
  timestamp?: string | Date;
  searchParams?: URLSearchParams;
}

export interface BrowserHarnessResult extends InitializeResult {
  runtime: RuntimeOverrides;
  report: Record<string, unknown>;
  reportInfo: SavedReportInfo;
}

export interface BrowserSuiteResult extends SuiteSummary {
  modelId?: string;
  metrics?: Record<string, unknown>;
  output?: string | DiffusionOutput | null;
  deviceInfo?: Record<string, unknown> | null;
  memoryStats?: ReturnType<InferencePipeline['getMemoryStats']> | null;
  pipeline?: InferencePipeline | DiffusionPipeline | EnergyPipeline | null;
  report: Record<string, unknown>;
  reportInfo: SavedReportInfo;
}

export interface BrowserManifestRun extends BrowserSuiteOptions {
  label?: string;
  runtimeConfigUrl?: string | null;
  runtimeConfig?: Record<string, unknown> | null;
}

export interface BrowserManifest {
  defaults?: BrowserManifestRun;
  runs: BrowserManifestRun[];
  reportModelId?: string;
  id?: string;
  report?: Record<string, unknown> | null;
}

export interface BrowserManifestResult {
  results: BrowserSuiteResult[];
  summary: {
    totalRuns: number;
    passedRuns: number;
    failedRuns: number;
    durationMs: number;
  };
  report: Record<string, unknown>;
  reportInfo: SavedReportInfo | null;
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

export declare function runBrowserSuite(
  options: BrowserSuiteOptions
): Promise<BrowserSuiteResult>;

export declare function runBrowserManifest(
  manifest: BrowserManifest,
  options?: RuntimeConfigLoadOptions & {
    saveReport?: boolean;
    timestamp?: string | Date;
    onProgress?: (progress: { index: number; total: number; label: string }) => void;
  }
): Promise<BrowserManifestResult>;
