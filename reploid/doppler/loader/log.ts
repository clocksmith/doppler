/**
 * Loader logging utility with trace support.
 *
 * Default: Show shard sources and layer timing
 * Trace:   Show tensor-level details (enabled via setTrace(true) or DOPPLER_TRACE env)
 *
 * @module loader/log
 */

// ============================================================================
// State
// ============================================================================

let traceEnabled = false;

// Check for browser/node environment
const getEnvTrace = (): boolean => {
  // Browser: check URL param
  if (typeof window !== 'undefined' && window.location) {
    const params = new URLSearchParams(window.location.search);
    return params.has('trace') || params.has('DOPPLER_TRACE');
  }
  // Node: check env var
  if (typeof process !== 'undefined' && process.env) {
    return process.env.DOPPLER_TRACE === '1' || process.env.DOPPLER_TRACE === 'true';
  }
  return false;
};

// Initialize from environment
traceEnabled = getEnvTrace();

// ============================================================================
// API
// ============================================================================

/**
 * Enable or disable trace-level logging
 */
export function setTrace(enabled: boolean): void {
  traceEnabled = enabled;
}

/**
 * Check if trace logging is enabled
 */
export function isTraceEnabled(): boolean {
  return traceEnabled;
}

/**
 * Standard log (always shown)
 * Use for: phase starts, completions, timing, shard sources
 */
export function log(message: string): void {
  console.log(`[Loader] ${message}`);
}

/**
 * Trace log (only when trace enabled)
 * Use for: tensor details, dequant ops, buffer sizes, debug info
 */
export function trace(message: string): void {
  if (traceEnabled) {
    console.log(`[Loader:trace] ${message}`);
  }
}

/**
 * Warning (always shown)
 */
export function warn(message: string): void {
  console.warn(`[Loader] ${message}`);
}
