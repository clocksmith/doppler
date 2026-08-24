import type { DebugCategory } from './config.js';

export declare const DEBUG_PROFILES: {
  quick: Partial<Record<DebugCategory, boolean>>;
  layers: Partial<Record<DebugCategory, boolean>>;
  attention: Partial<Record<DebugCategory, boolean>>;
  full: Partial<Record<DebugCategory, boolean>>;
  perf: Partial<Record<DebugCategory, boolean>>;
  kernelStep: Partial<Record<DebugCategory, boolean>>;
};
