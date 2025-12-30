/**
 * Unified logging system for DOPPLER loader.
 *
 * Log levels (progressive - each includes all levels below it):
 *   silent  - nothing
 *   error   - errors only
 *   info    - phase starts/ends, totals (default for test/bench)
 *   verbose - per-shard source, per-layer timing (default for debug)
 *   trace   - tensor shapes, dequant ops, buffer details
 *
 * Configuration priority:
 *   1. Explicit setLogLevel() call
 *   2. URL param: ?log=verbose or ?trace (sets trace)
 *   3. CLI flag: --verbose, --trace, --quiet
 *   4. Command default: test=info, bench=info, debug=verbose
 *
 * @module loader/log
 */

// ============================================================================
// Types
// ============================================================================

export type LogLevel = 'silent' | 'error' | 'info' | 'verbose' | 'trace';

const LEVEL_PRIORITY: Record<LogLevel, number> = {
  silent: 0,
  error: 1,
  info: 2,
  verbose: 3,
  trace: 4,
};

// ============================================================================
// State
// ============================================================================

let currentLevel: LogLevel = 'info';  // Default

// ============================================================================
// Auto-detect from environment
// ============================================================================

function detectLevel(): LogLevel {
  // Browser: check URL params
  if (typeof window !== 'undefined' && window.location) {
    const params = new URLSearchParams(window.location.search);

    // Explicit log level
    const logParam = params.get('log');
    if (logParam && logParam in LEVEL_PRIORITY) {
      return logParam as LogLevel;
    }

    // Shorthand flags
    if (params.has('trace') || params.has('DOPPLER_TRACE')) return 'trace';
    if (params.has('verbose')) return 'verbose';
    if (params.has('quiet')) return 'silent';
  }

  // Node: check env vars
  if (typeof process !== 'undefined' && process.env) {
    const envLog = process.env.DOPPLER_LOG;
    if (envLog && envLog in LEVEL_PRIORITY) {
      return envLog as LogLevel;
    }
    if (process.env.DOPPLER_TRACE === '1') return 'trace';
    if (process.env.DOPPLER_VERBOSE === '1') return 'verbose';
    if (process.env.DOPPLER_QUIET === '1') return 'silent';
  }

  return 'info';  // Default
}

// Initialize from environment
currentLevel = detectLevel();

// ============================================================================
// API
// ============================================================================

/**
 * Set the log level explicitly
 */
export function setLogLevel(level: LogLevel): void {
  currentLevel = level;
}

/**
 * Get current log level
 */
export function getLogLevel(): LogLevel {
  return currentLevel;
}

/**
 * Check if a level would be logged
 */
export function shouldLog(level: LogLevel): boolean {
  return LEVEL_PRIORITY[level] <= LEVEL_PRIORITY[currentLevel];
}

// ============================================================================
// Log Functions
// ============================================================================

const PREFIX = '[Loader]';

/**
 * Error log (level: error+)
 */
export function error(message: string, ...args: unknown[]): void {
  if (shouldLog('error')) {
    console.error(`${PREFIX} ${message}`, ...args);
  }
}

/**
 * Warning log (level: error+)
 */
export function warn(message: string, ...args: unknown[]): void {
  if (shouldLog('error')) {
    console.warn(`${PREFIX} ${message}`, ...args);
  }
}

/**
 * Info log (level: info+)
 * Use for: phase starts/ends, model loaded, totals
 */
export function log(message: string, ...args: unknown[]): void {
  if (shouldLog('info')) {
    console.log(`${PREFIX} ${message}`, ...args);
  }
}

/**
 * Verbose log (level: verbose+)
 * Use for: per-shard source, per-layer timing
 */
export function verbose(message: string, ...args: unknown[]): void {
  if (shouldLog('verbose')) {
    console.log(`${PREFIX} ${message}`, ...args);
  }
}

/**
 * Trace log (level: trace only)
 * Use for: tensor shapes, dequant details, buffer sizes
 */
export function trace(message: string, ...args: unknown[]): void {
  if (shouldLog('trace')) {
    console.log(`${PREFIX}:trace ${message}`, ...args);
  }
}
