export interface LogLevels {
  DEBUG: 0;
  VERBOSE: 1;
  INFO: 2;
  WARN: 3;
  ERROR: 4;
  SILENT: 5;
}

export type LogLevelName = 'debug' | 'verbose' | 'info' | 'warn' | 'error' | 'silent';
export type TraceCategory = 'loader' | 'kernels' | 'logits' | 'embed' | 'attn' | 'ffn' | 'kv' | 'sample' | 'buffers' | 'perf' | 'energy';

export interface TraceOptions {
  layers?: number[];
  maxDecodeSteps?: number;
  breakOnAnomaly?: boolean;
}

export interface DebugConfig {
  logLevel?: {
    defaultLogLevel?: LogLevelName;
  };
  trace?: {
    enabled?: boolean;
    categories?: TraceCategory[];
    layers?: number[];
    maxDecodeSteps?: number;
  };
  logHistory?: {
    maxLogHistoryEntries?: number;
  };
}


export interface LogHistoryEntry {
  time: number;
  perfTime: number;
  level: string;
  module: string;
  message: string;
  data?: unknown;
}

export const LOG_LEVELS: LogLevels;
export const TRACE_CATEGORIES: readonly TraceCategory[];

export let currentLogLevel: number;
export let enabledModules: Set<string>;
export let disabledModules: Set<string>;
export let logHistory: LogHistoryEntry[];
export let gpuDevice: GPUDevice | null;
export let enabledTraceCategories: Set<TraceCategory>;
export let traceLayerFilter: number[];
export let traceDecodeStep: number;
export let traceMaxDecodeSteps: number;
export let traceBreakOnAnomaly: boolean;

export function setLogLevel(level: LogLevelName): void;
export function getLogLevel(): LogLevelName;
export function setTrace(categories: string | TraceCategory[] | false, options?: TraceOptions): void;
export function applyDebugConfig(config: DebugConfig): void;
export function getLogHistoryLimit(): number;
export function getTrace(): TraceCategory[];
export function isTraceEnabled(category: TraceCategory, layerIdx?: number): boolean;
export function incrementDecodeStep(): number;
export function resetDecodeStep(): void;
export function getDecodeStep(): number;
export function shouldBreakOnAnomaly(): boolean;
export function setSilentMode(enabled: boolean): void;
export function isSilentMode(): boolean;
/** @deprecated Use setSilentMode instead */
export function setBenchmarkMode(enabled: boolean): void;
/** @deprecated Use isSilentMode instead */
export function isBenchmarkMode(): boolean;
export function enableModules(...modules: string[]): void;
export function disableModules(...modules: string[]): void;
export function resetModuleFilters(): void;
export function setGPUDevice(device: GPUDevice | null): void;
