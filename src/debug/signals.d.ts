/**
 * DOPPLER Debug Module - Completion Signals
 *
 * Standard completion signal prefixes for CLI/automation detection.
 *
 * Usage:
 *   console.log(`${SIGNALS.DONE} ${JSON.stringify({status: 'success', elapsed: 1234})}`);
 *   console.log(`${SIGNALS.RESULT} ${JSON.stringify(benchmarkData)}`);
 *   console.log(`${SIGNALS.ERROR} ${JSON.stringify({error: 'message'})}`);
 *
 * Detection (CLI/Puppeteer):
 *   if (text.startsWith('[DOPPLER:DONE]')) { ... }
 *
 * @module debug/signals
 */

/**
 * Standard completion signal prefixes for CLI/automation detection.
 */
export declare const SIGNALS: {
  /** Task completed (success or error) - always emitted at end */
  readonly DONE: '[DOPPLER:DONE]';
  /** Full result payload (JSON) - emitted before DONE for data extraction */
  readonly RESULT: '[DOPPLER:RESULT]';
  /** Error occurred - can be emitted before DONE */
  readonly ERROR: '[DOPPLER:ERROR]';
  /** Progress update (optional) */
  readonly PROGRESS: '[DOPPLER:PROGRESS]';
};

export type SignalType = keyof typeof SIGNALS;

/**
 * Completion payload for DONE signal.
 */
export interface DonePayload {
  status: 'success' | 'error';
  elapsed: number;
  tokens?: number;
  tokensPerSecond?: number;
  error?: string;
}

/**
 * Emit a completion signal to console.
 * This is the standard way to signal task completion for CLI detection.
 */
export declare function signalDone(payload: DonePayload): void;

/**
 * Emit a result signal with full data payload.
 */
export declare function signalResult(data: Record<string, unknown>): void;

/**
 * Emit an error signal.
 */
export declare function signalError(error: string, details?: Record<string, unknown>): void;

/**
 * Emit a progress signal.
 */
export declare function signalProgress(percent: number, message?: string): void;
