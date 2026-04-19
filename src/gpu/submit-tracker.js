

import { trackSubmit } from './perf-guards.js';
import { log, trace } from '../debug/index.js';


let TRACK_SUBMITS = false;


let submitCount = 0;

let submitTimes = [];
let totalSubmitMs = 0;
let maxSubmitMs = 0;
let minSubmitMs = Infinity;

let submitSources = new Map();




let currentPhase = 'other';

const WRAPPED_QUEUE_SENTINEL = Symbol.for('doppler.submitTrackerWrapped');


const phaseStats = {
  prefill: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
  decode: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
  other: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() },
};


export function setTrackSubmits(enabled) {
  TRACK_SUBMITS = enabled;
  if (enabled) {
    resetSubmitStats();
    log.debug('SubmitTracker', 'Enabled');
  } else {
    log.debug('SubmitTracker', 'Disabled');
  }
}


export function resetSubmitStats() {
  submitCount = 0;
  submitTimes = [];
  totalSubmitMs = 0;
  maxSubmitMs = 0;
  minSubmitMs = Infinity;
  submitSources = new Map();
  currentPhase = 'other';

  for (const phase of  (['prefill', 'decode', 'other'])) {
    phaseStats[phase] = { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: new Map() };
  }
}


function recordSubmit(durationMs, source) {
  if (!TRACK_SUBMITS) return;

  submitCount++;
  submitTimes.push(durationMs);
  totalSubmitMs += durationMs;
  maxSubmitMs = Math.max(maxSubmitMs, durationMs);
  minSubmitMs = Math.min(minSubmitMs, durationMs);

  if (source) {
    submitSources.set(source, (submitSources.get(source) || 0) + 1);
  }

  const ps = phaseStats[currentPhase];
  ps.count++;
  ps.times.push(durationMs);
  ps.totalMs += durationMs;
  ps.maxMs = Math.max(ps.maxMs, durationMs);
  ps.minMs = Math.min(ps.minMs, durationMs);

  if (source) {
    ps.sources.set(source, (ps.sources.get(source) || 0) + 1);
  }
}


export function getSubmitStats() {
  return {
    count: submitCount,
    totalMs: totalSubmitMs,
    avgMs: submitCount > 0 ? totalSubmitMs / submitCount : 0,
    maxMs: maxSubmitMs,
    minMs: minSubmitMs === Infinity ? 0 : minSubmitMs,
    timestamps: [...submitTimes],
    bySource: new Map(submitSources),
  };
}


function getPhaseSubmitStats(phase) {
  const ps = phaseStats[phase];
  return {
    count: ps.count,
    totalMs: ps.totalMs,
    avgMs: ps.count > 0 ? ps.totalMs / ps.count : 0,
    maxMs: ps.maxMs,
    minMs: ps.minMs === Infinity ? 0 : ps.minMs,
    timestamps: [...ps.times],
    bySource: new Map(ps.sources),
  };
}


export function logSubmitStats(label = 'Forward pass') {
  const stats = getSubmitStats();
  trace.perf(
    `SubmitTracker ${label}: ${stats.count} submits, ` +
    `total=${stats.totalMs.toFixed(2)}ms, ` +
    `avg=${stats.avgMs.toFixed(3)}ms, ` +
    `range=[${stats.minMs.toFixed(3)}-${stats.maxMs.toFixed(3)}ms]`
  );

  if (stats.bySource && stats.bySource.size > 0) {
    trace.perf('SubmitTracker: Submits by source:');
    const sorted = Array.from(stats.bySource.entries()).sort((a, b) => b[1] - a[1]);
    for (const [source, count] of sorted) {
      const pct = ((count / stats.count) * 100).toFixed(1);
      trace.perf(`  ${source}: ${count} (${pct}%)`);
    }
  }
}


function extractSourceFromStack() {
  const stack = new Error().stack;
  if (!stack) return 'unknown';

  const lines = stack.split('\n');
  // Skip first 3 lines: Error, extractSourceFromStack, queue.submit wrapper
  for (let i = 3; i < lines.length; i++) {
    const line = lines[i];
    // Match file:line pattern in stack trace
    // Example: "at functionName (http://localhost:8080/path/to/file.ts:123:45)"
    const match = line.match(/\/([^\/]+\.(?:js|ts)):(\d+):/);
    if (match) {
      return `${match[1]}:${match[2]}`;
    }
  }
  return 'unknown';
}


export function wrapQueueForTracking(queue) {
  if (!queue || queue[WRAPPED_QUEUE_SENTINEL] === true) {
    return queue;
  }

  const originalSubmit = queue.submit.bind(queue);

   (queue).submit = function( commandBuffers) {
    const start = TRACK_SUBMITS ? performance.now() : 0;
    const result = originalSubmit(commandBuffers);
    trackSubmit();

    if (!TRACK_SUBMITS) {
      return result;
    }

    const duration = performance.now() - start;
    recordSubmit(duration, extractSourceFromStack());
    return result;
  };

  Object.defineProperty(queue, WRAPPED_QUEUE_SENTINEL, {
    value: true,
    configurable: true,
    enumerable: false,
    writable: false,
  });

  return queue;
}


