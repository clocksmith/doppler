/**
 * Logging functions for pipeline debug tracing.
 *
 * Provides category-specific logging for each pipeline stage:
 * embedding, layers, attention, FFN, KV cache, logits, sampling, I/O, and performance.
 *
 * Log format: [CATEGORY] message
 * This enables post-filtering: grep -E "^\[LAYER\]|\[ATTN\]"
 *
 * @module inference/pipelines/text/debug-utils/logging
 */

// ============================================================================
// Logging Functions
// ============================================================================

/**
 * Log embedding info.
 */
export function logEmbed(
  tokenIds: number[],
  info: { maxAbs?: number; nonZero?: number; total?: number; sample?: number[] }
): void;

/**
 * Log layer entry/exit.
 */
export function logLayer(
  layerIdx: number,
  phase: 'enter' | 'exit',
  isPrefill: boolean,
  info: { numTokens?: number; maxAbs?: number; sample?: number[] }
): void;

/**
 * Log attention details.
 */
export function logAttn(
  layerIdx: number,
  isPrefill: boolean,
  info: {
    numTokens: number;
    kvLen: number;
    startPos?: number;
    maxAbsQ?: number;
    maxAbsK?: number;
    maxAbsV?: number;
    maxAbsOut?: number;
  }
): void;

/**
 * Log FFN details.
 */
export function logFFN(
  layerIdx: number,
  info: { maxAbsGate?: number; maxAbsUp?: number; maxAbsOut?: number }
): void;

/**
 * Log KV cache operations.
 */
export function logKV(
  layerIdx: number,
  op: 'write' | 'read' | 'init' | 'clear',
  info: { seqLen?: number; kvLen?: number; startPos?: number }
): void;

/**
 * Log logits computation.
 */
export function logLogits(
  phase: 'prefill' | 'decode',
  info: {
    min: number;
    max: number;
    topK?: Array<{ token: number | string; prob: number; text?: string }>;
  }
): void;

/**
 * Log sampling decision.
 */
export function logSample(
  tokenId: number,
  tokenText: string,
  info: { prob?: number; temperature?: number; topK?: number }
): void;

/**
 * Log GPU buffer I/O.
 */
export function logIO(
  op: 'read' | 'write' | 'copy',
  label: string,
  bytes: number
): void;

/**
 * Log performance timing.
 */
export function logPerf(
  phase: string,
  ms: number,
  extra?: Record<string, number | string>
): void;
