/**
 * Debug Module Global State
 *
 * @module debug/state
 */

import { LOG_LEVELS, type LogLevelValue, type TraceCategory, type LogEntry } from './types.js';

// Global state variables
export let currentLogLevel: LogLevelValue = LOG_LEVELS.INFO;
export let enabledModules = new Set<string>();
export let disabledModules = new Set<string>();
export let logHistory: LogEntry[] = [];

// GPU device reference for tensor inspection
export let gpuDevice: GPUDevice | null = null;

// Trace categories state
export let enabledTraceCategories = new Set<TraceCategory>();
export let traceLayerFilter: number[] = [];  // Empty = all layers
export let traceDecodeStep = 0;
export let traceMaxDecodeSteps = 0;  // 0 = unlimited
export let traceBreakOnAnomaly = false;

// Benchmark mode state
export let benchmarkMode = false;

// Helpers to update state (needed since we can't export setters for 'let' variables easily across modules)
export function setCurrentLogLevel(level: LogLevelValue): void {
  currentLogLevel = level;
}

export function setEnabledModules(modules: Set<string>): void {
  enabledModules = modules;
}

export function setDisabledModules(modules: Set<string>): void {
  disabledModules = modules;
}

export function setGPUDeviceRef(device: GPUDevice | null): void {
  gpuDevice = device;
}

export function setTraceLayerFilter(layers: number[]): void {
  traceLayerFilter = layers;
}

export function setTraceMaxDecodeSteps(steps: number): void {
  traceMaxDecodeSteps = steps;
}

export function setTraceBreakOnAnomaly(breakOn: boolean): void {
  traceBreakOnAnomaly = breakOn;
}

export function incrementTraceDecodeStep(): number {
  return ++traceDecodeStep;
}

export function resetTraceDecodeStep(): void {
  traceDecodeStep = 0;
}

export function setBenchmarkModeState(enabled: boolean): void {
  benchmarkMode = enabled;
}

export function clearHistory(): void {
  logHistory = [];
}

export function pushHistory(entry: LogEntry): void {
  logHistory.push(entry);
}

export function shiftHistory(): LogEntry | undefined {
  return logHistory.shift();
}
