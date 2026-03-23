import { log } from '../debug/index.js';

// Benchmark state accumulates across sequential runs unless explicitly reset.
let benchState = {
  runCount: 0,
  lastRunTimestamp: null,
  accumulatedMetrics: [],
  warmupComplete: false,
};

/**
 * Reset benchmark state to initial values.
 * Called at the start of each run to prevent state leaking between sequential runs.
 */
export function resetBenchState() {
  benchState = {
    runCount: 0,
    lastRunTimestamp: null,
    accumulatedMetrics: [],
    warmupComplete: false,
  };
  log.debug('bench-runner', 'Benchmark state reset');
}

/**
 * Get current benchmark state (read-only snapshot).
 * @returns {object}
 */
export function getBenchState() {
  return { ...benchState };
}

/**
 * Run a benchmark pass, resetting state at the start to avoid contamination
 * from prior sequential runs.
 *
 * @param {object} options
 * @param {Function} options.execute - The benchmark function to run.
 * @param {string} [options.label] - Optional label for logging.
 * @returns {Promise<object>} The benchmark result.
 */
export async function runBench(options = {}) {
  resetBenchState();

  const { execute, label } = options;
  if (typeof execute !== 'function') {
    throw new Error('bench-runner: options.execute must be a function.');
  }

  const runLabel = label || `run-${Date.now()}`;
  log.debug('bench-runner', `Starting benchmark: ${runLabel}`);

  benchState.runCount += 1;
  benchState.lastRunTimestamp = Date.now();

  const result = await execute();

  if (result && typeof result === 'object') {
    benchState.accumulatedMetrics.push(result);
  }

  log.debug('bench-runner', `Benchmark complete: ${runLabel}`);
  return result;
}
