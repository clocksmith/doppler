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

export interface DiagnosticObserver { observe(event: Readonly<Record<string, unknown>>): void; }
export declare function resolveDiagnosticObserver(observer?: DiagnosticObserver | null): DiagnosticObserver | null;
/** Used only inside the serialized pipeline lease. */
export declare function enterDiagnosticObserver(observer: DiagnosticObserver | null): () => void;
export declare function emitDiagnostic(event: Record<string, unknown>): boolean;
