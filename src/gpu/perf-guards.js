

import { log, trace } from '../debug/index.js';
import { DEFAULT_PERF_GUARDS_CONFIG } from '../config/schema/index.js';


let config = { ...DEFAULT_PERF_GUARDS_CONFIG };


let counters = {
  submits: 0,
  allocations: 0,
  readbacks: 0,
  startTime: 0,
};


export function configurePerfGuards(newConfig) {
  config = { ...config, ...newConfig };
}


export function getPerfConfig() {
  return config;
}


export function resetPerfCounters() {
  counters = {
    submits: 0,
    allocations: 0,
    readbacks: 0,
    startTime: performance.now(),
  };
}


export function getPerfCounters() {
  return counters;
}


export function trackSubmit() {
  if (config.trackSubmitCount) {
    counters.submits++;
    if (config.logExpensiveOps) {
      trace.perf(`PerfGuard: Submit #${counters.submits}`);
    }
  }
}


export function trackAllocation(size, label) {
  if (config.trackAllocations) {
    counters.allocations++;
    if (config.logExpensiveOps) {
      trace.buffers(`PerfGuard: Allocation #${counters.allocations}: ${size} bytes (${label || 'unlabeled'})`);
    }
  }
}


export function allowReadback(reason, count = 1) {
  if (!config.allowGPUReadback) {
    const message = `PerfGuard: GPU readback blocked: ${reason || 'unknown reason'}`;
    if (config.strictMode) {
      throw new Error(message);
    }
    if (config.logExpensiveOps) {
      log.warn('PerfGuard', message);
    }
    return false;
  }

  if (config.trackSubmitCount) {
    const increment = Number.isFinite(count) ? Math.max(0, Math.floor(count)) : 1;
    if (increment > 0) {
      counters.readbacks += increment;
    }
    if (config.logExpensiveOps) {
      trace.perf(`PerfGuard: Readback #${counters.readbacks}: ${reason || 'unknown'} (${count})`);
    }
  }

  return true;
}


export function getPerfSummary() {
  const elapsed = performance.now() - counters.startTime;
  return [
    `Performance Summary (${elapsed.toFixed(1)}ms):`,
    `  Submits: ${counters.submits}`,
    `  Allocations: ${counters.allocations}`,
    `  Readbacks: ${counters.readbacks}`,
  ].join('\n');
}


export function logPerfSummary() {
  trace.perf(getPerfSummary());
}


export function enableProductionMode() {
  configurePerfGuards({
    allowGPUReadback: false,
    trackSubmitCount: false,
    trackAllocations: false,
    logExpensiveOps: false,
    strictMode: true,
  });
}


export function enableDebugMode() {
  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: true,
    trackAllocations: true,
    logExpensiveOps: true,
    strictMode: false,
  });
}


export function enableBenchmarkMode() {
  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: true,
    trackAllocations: true,
    logExpensiveOps: false,
    strictMode: false,
  });
}
