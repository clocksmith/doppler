export type ToolingCommand = 'convert' | 'debug' | 'bench' | 'test-model';
export type ToolingSurface = 'browser' | 'node';
export type ToolingSuite = 'kernels' | 'inference' | 'bench' | 'debug' | 'diffusion' | 'energy';
export type ToolingIntent = 'verify' | 'investigate' | 'calibrate' | null;

export interface ToolingCommandRequestInput {
  command: ToolingCommand;
  suite?: ToolingSuite;
  modelId?: string;
  modelUrl?: string;
  runtimePreset?: string;
  runtimeConfigUrl?: string;
  runtimeConfig?: Record<string, unknown>;
  inputDir?: string;
  outputDir?: string;
  convertPayload?: Record<string, unknown>;
  captureOutput?: boolean;
  keepPipeline?: boolean;
  report?: Record<string, unknown> | null;
  timestamp?: string | Date | null;
  searchParams?: URLSearchParams | null;
}

export interface ToolingCommandRequest {
  command: ToolingCommand;
  suite: ToolingSuite | null;
  intent: ToolingIntent;
  modelId: string | null;
  modelUrl: string | null;
  runtimePreset: string | null;
  runtimeConfigUrl: string | null;
  runtimeConfig: Record<string, unknown> | null;
  inputDir: string | null;
  outputDir: string | null;
  convertPayload: Record<string, unknown> | null;
  captureOutput: boolean;
  keepPipeline: boolean;
  report: Record<string, unknown> | null;
  timestamp: string | Date | null;
  searchParams: URLSearchParams | null;
}

export declare const TOOLING_COMMANDS: readonly ToolingCommand[];
export declare const TOOLING_SURFACES: readonly ToolingSurface[];
export declare const TOOLING_SUITES: readonly ToolingSuite[];

export declare function normalizeToolingCommandRequest(
  input: ToolingCommandRequestInput
): ToolingCommandRequest;

export declare function buildRuntimeContractPatch(
  commandRequest: ToolingCommandRequestInput
): {
  shared: {
    harness: { mode: ToolingSuite; modelId: string | null };
    tooling: { intent: Exclude<ToolingIntent, null> };
  };
} | null;

export declare function ensureCommandSupportedOnSurface(
  commandRequest: ToolingCommandRequestInput,
  surface: ToolingSurface
): { request: ToolingCommandRequest; surface: ToolingSurface };
