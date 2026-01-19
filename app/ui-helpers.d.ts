/**
 * ui-helpers.d.ts - UI helper utilities
 *
 * @module app/ui-helpers
 */

import type { CapabilitiesState } from './capabilities-detector.js';
import type { GPUElements, StatsElements } from './app.js';

export type StatusState = string;

export interface GPUInfoData {
  deviceName: string;
  bufferLimitBytes: number;
  isUnifiedMemory: boolean;
  systemMemoryGB?: number | null;
  hasF16: boolean;
  hasSubgroups: boolean;
  hasTimestamps: boolean;
}

export interface StatsData {
  tps?: number | string;
  memoryMB?: number | string;
  gpuBuffers?: number | string;
  kvSeqLen?: number;
  kvMaxLen?: number;
}

export declare function setStatus(
  dotEl: HTMLElement | null,
  textEl: HTMLElement | null,
  state: StatusState,
  text: string
): void;

export declare function updateCapabilitiesUI(
  listEl: HTMLElement | null,
  capabilities: CapabilitiesState
): void;

export declare function populateGPUInfo(
  elements: GPUElements,
  data: GPUInfoData
): void;

export declare function updateStats(
  elements: StatsElements,
  stats: StatsData
): void;

export declare function resetStats(elements: StatsElements): void;

export declare function showError(
  modalEl: HTMLElement | null,
  messageEl: HTMLElement | null,
  closeBtn: HTMLElement | null,
  message: string
): void;

export declare function formatBytes(bytes: number): string;

export declare function clampInputValue(
  input: HTMLInputElement,
  min: number,
  max: number
): void;
