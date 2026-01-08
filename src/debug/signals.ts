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
export const SIGNALS = {
  /** Task completed (success or error) - always emitted at end */
  DONE: '[DOPPLER:DONE]',
  /** Full result payload (JSON) - emitted before DONE for data extraction */
  RESULT: '[DOPPLER:RESULT]',
  /** Error occurred - can be emitted before DONE */
  ERROR: '[DOPPLER:ERROR]',
  /** Progress update (optional) */
  PROGRESS: '[DOPPLER:PROGRESS]',
} as const;

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
export function signalDone(payload: DonePayload): void {
  console.log(`${SIGNALS.DONE} ${JSON.stringify(payload)}`);
}

/**
 * Emit a result signal with full data payload.
 */
export function signalResult(data: Record<string, unknown>): void {
  console.log(`${SIGNALS.RESULT} ${JSON.stringify(data)}`);
}

/**
 * Emit an error signal.
 */
export function signalError(error: string, details?: Record<string, unknown>): void {
  console.log(`${SIGNALS.ERROR} ${JSON.stringify({ error, ...details })}`);
}

/**
 * Emit a progress signal.
 */
export function signalProgress(percent: number, message?: string): void {
  console.log(`${SIGNALS.PROGRESS} ${JSON.stringify({ percent, message })}`);
}
