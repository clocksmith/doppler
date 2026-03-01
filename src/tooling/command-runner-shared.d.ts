import type { RuntimeConfigLoadOptions } from '../inference/browser-harness.js';
import type { ToolingCommandRequest } from './command-api.js';

export interface RuntimeBridge {
  applyRuntimePreset: (
    runtimePreset: string,
    options?: RuntimeConfigLoadOptions
  ) => Promise<void>;
  applyRuntimeConfigFromUrl: (
    runtimeConfigUrl: string,
    options?: RuntimeConfigLoadOptions
  ) => Promise<void>;
  getRuntimeConfig: () => Record<string, unknown>;
  setRuntimeConfig: (runtimeConfig: Record<string, unknown> | null) => void;
  getActiveKernelPath?: () => unknown;
  getActiveKernelPathSource?: () => string;
  getActiveKernelPathPolicy?: () => Record<string, unknown> | null;
  setActiveKernelPath?: (path: unknown, source?: string, policy?: Record<string, unknown> | null) => void;
}

export declare function applyRuntimeInputs(
  request: ToolingCommandRequest,
  runtimeBridge: RuntimeBridge,
  options?: RuntimeConfigLoadOptions
): Promise<void>;

export declare function buildSuiteOptions(
  request: ToolingCommandRequest,
  surface?: string | null
): {
  suite: ToolingCommandRequest['suite'];
  command: ToolingCommandRequest['command'];
  surface: string | null;
  modelId?: string;
  cacheMode: ToolingCommandRequest['cacheMode'];
  loadMode: 'opfs' | 'http' | 'memory' | null;
  modelUrl?: string;
  runtimePreset: string | null;
  captureOutput: boolean;
  keepPipeline: boolean;
  report?: Record<string, unknown>;
  timestamp?: string | Date;
  searchParams?: URLSearchParams;
};
