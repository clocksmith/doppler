/**
 * memory-monitor.d.ts - UI memory stats polling
 *
 * @module app/memory-monitor
 */

import type { MemoryElements } from './app.js';

export interface PipelineMemoryStats {
  kvCache?: {
    allocated?: number | null;
  } | null;
}

export interface MemoryMonitorConfig {
  estimatedSystemMemoryBytes?: number;
  gpuBufferLimitBytes?: number;
  isUnifiedMemory?: boolean;
  getPipelineMemoryStats?: (() => PipelineMemoryStats | null) | null;
  pollIntervalMs?: number;
}

export declare class MemoryMonitor {
  constructor(elements: MemoryElements);

  configure(config: MemoryMonitorConfig): void;
  start(): void;
  stop(): void;
  update(): void;
  formatBytes(bytes: number): string;
}
