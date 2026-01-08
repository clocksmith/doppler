/**
 * DOPPLER Debug Module - Core Logging Interface
 *
 * Provides structured logging with level filtering and history tracking.
 *
 * @module debug/log
 */

// ============================================================================
// Logging Interface
// ============================================================================

/**
 * Main logging interface.
 */
export declare const log: {
  /**
   * Debug level logging (most verbose).
   */
  debug(module: string, message: string, data?: unknown): void;

  /**
   * Verbose level logging (detailed operational info).
   */
  verbose(module: string, message: string, data?: unknown): void;

  /**
   * Info level logging (normal operations).
   */
  info(module: string, message: string, data?: unknown): void;

  /**
   * Warning level logging.
   */
  warn(module: string, message: string, data?: unknown): void;

  /**
   * Error level logging.
   */
  error(module: string, message: string, data?: unknown): void;

  /**
   * Always log regardless of level (for critical messages).
   */
  always(module: string, message: string, data?: unknown): void;
};
