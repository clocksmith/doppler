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
}

export declare function applyRuntimeInputs(
  request: ToolingCommandRequest,
  runtimeBridge: RuntimeBridge,
  options?: RuntimeConfigLoadOptions
): Promise<void>;

export declare function buildSuiteOptions(
  request: ToolingCommandRequest
): {
  suite: ToolingCommandRequest['suite'];
  modelId?: string;
  modelUrl?: string;
  runtimePreset: string | null;
  captureOutput: boolean;
  keepPipeline: boolean;
  report?: Record<string, unknown>;
  timestamp?: string | Date;
  searchParams?: URLSearchParams;
};
