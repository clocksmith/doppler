/**
 * Logging functions for pipeline debug tracing.
 *
 * Provides category-specific logging for each pipeline stage:
 * embedding, layers, attention, FFN, KV cache, logits, sampling, I/O, and performance.
 *
 * Log format: [CATEGORY] message
 * This enables post-filtering: grep -E "^\[LAYER\]|\[ATTN\]"
 *
 * @module inference/pipeline/debug-utils/logging
 */

import { log } from '../../../debug/index.js';
import {
  isEnabled,
  formatTag,
  getDecodeStep,
  getMaxAbsThreshold,
} from './config.js';

// ============================================================================
// Logging Functions
// ============================================================================

/**
 * Log embedding info.
 * @param {number[]} tokenIds
 * @param {{ maxAbs?: number; nonZero?: number; total?: number; sample?: number[] }} info
 * @returns {void}
 */
export function logEmbed(tokenIds, info) {
  if (!isEnabled('embed')) return;

  const tag = formatTag('embed');
  const tokens = tokenIds.length > 3
    ? `[${tokenIds.slice(0, 3).join(',')},...] (${tokenIds.length} total)`
    : `[${tokenIds.join(',')}]`;

  let msg = `${tag} tokens=${tokens}`;
  if (info.maxAbs !== undefined) msg += ` maxAbs=${info.maxAbs.toFixed(2)}`;
  if (info.nonZero !== undefined) msg += ` nonZero=${info.nonZero}/${info.total}`;
  if (info.sample?.length) msg += ` sample=[${info.sample.map(v => v.toFixed(3)).join(',')}]`;

  log.debug('Debug', msg);
}

/**
 * Log layer entry/exit.
 * @param {number} layerIdx
 * @param {'enter' | 'exit'} phase
 * @param {boolean} isPrefill
 * @param {{ numTokens?: number; maxAbs?: number; sample?: number[] }} info
 * @returns {void}
 */
export function logLayer(layerIdx, phase, isPrefill, info) {
  if (!isEnabled('layer', layerIdx)) return;

  const tag = formatTag('layer', layerIdx);
  const decodeStep = getDecodeStep();
  const mode = isPrefill ? 'prefill' : `decode:${decodeStep}`;

  let msg = `${tag} ${phase} ${mode}`;
  if (info.numTokens !== undefined) msg += ` n=${info.numTokens}`;
  if (info.maxAbs !== undefined) msg += ` maxAbs=${info.maxAbs.toFixed(2)}`;
  if (info.sample?.length) msg += ` sample=[${info.sample.map(v => v.toFixed(3)).join(',')}]`;

  // Warn on explosion
  const maxAbsThreshold = getMaxAbsThreshold();
  if (info.maxAbs !== undefined && info.maxAbs > maxAbsThreshold) {
    msg += ` ☡ EXPLOSION`;
  }

  log.debug('Debug', msg);
}

/**
 * Log attention details.
 * @param {number} layerIdx
 * @param {boolean} isPrefill
 * @param {{ numTokens: number; kvLen: number; startPos?: number; maxAbsQ?: number; maxAbsK?: number; maxAbsV?: number; maxAbsOut?: number }} info
 * @returns {void}
 */
export function logAttn(layerIdx, isPrefill, info) {
  if (!isEnabled('attn', layerIdx)) return;

  const tag = formatTag('attn', layerIdx);
  const decodeStep = getDecodeStep();
  const mode = isPrefill ? 'prefill' : `decode:${decodeStep}`;

  let msg = `${tag} ${mode} n=${info.numTokens} kvLen=${info.kvLen}`;
  if (info.startPos !== undefined) msg += ` startPos=${info.startPos}`;
  if (info.maxAbsQ !== undefined) msg += ` Q=${info.maxAbsQ.toFixed(1)}`;
  if (info.maxAbsK !== undefined) msg += ` K=${info.maxAbsK.toFixed(1)}`;
  if (info.maxAbsV !== undefined) msg += ` V=${info.maxAbsV.toFixed(1)}`;
  if (info.maxAbsOut !== undefined) msg += ` out=${info.maxAbsOut.toFixed(1)}`;

  log.debug('Debug', msg);
}

/**
 * Log FFN details.
 * @param {number} layerIdx
 * @param {{ maxAbsGate?: number; maxAbsUp?: number; maxAbsOut?: number }} info
 * @returns {void}
 */
export function logFFN(layerIdx, info) {
  if (!isEnabled('ffn', layerIdx)) return;

  const tag = formatTag('ffn', layerIdx);
  let msg = tag;
  if (info.maxAbsGate !== undefined) msg += ` gate=${info.maxAbsGate.toFixed(1)}`;
  if (info.maxAbsUp !== undefined) msg += ` up=${info.maxAbsUp.toFixed(1)}`;
  if (info.maxAbsOut !== undefined) msg += ` out=${info.maxAbsOut.toFixed(1)}`;

  log.debug('Debug', msg);
}

/**
 * Log KV cache operations.
 * @param {number} layerIdx
 * @param {'write' | 'read' | 'init' | 'clear'} op
 * @param {{ seqLen?: number; kvLen?: number; startPos?: number }} info
 * @returns {void}
 */
export function logKV(layerIdx, op, info) {
  if (!isEnabled('kv', layerIdx)) return;

  const tag = formatTag('kv', layerIdx);
  let msg = `${tag} ${op}`;
  if (info.seqLen !== undefined) msg += ` seqLen=${info.seqLen}`;
  if (info.kvLen !== undefined) msg += ` kvLen=${info.kvLen}`;
  if (info.startPos !== undefined) msg += ` startPos=${info.startPos}`;

  log.debug('Debug', msg);
}

/**
 * Log logits computation.
 * @param {'prefill' | 'decode'} phase
 * @param {{ min: number; max: number; topK?: Array<{ token: number | string; prob: number; text?: string }> }} info
 * @returns {void}
 */
export function logLogits(phase, info) {
  if (!isEnabled('logits')) return;

  const decodeStep = getDecodeStep();
  const tag = phase === 'prefill' ? '[LOGITS][PREFILL]' : `[LOGITS][S${decodeStep}]`;
  let msg = `${tag} min=${info.min.toFixed(2)} max=${info.max.toFixed(2)}`;

  if (info.topK?.length) {
    const topStr = info.topK
      .slice(0, 5)
      .map(t => `"${t.text || t.token}"(${(t.prob * 100).toFixed(1)}%)`)
      .join(', ');
    msg += ` top-5: ${topStr}`;
  }

  log.debug('Debug', msg);
}

/**
 * Log sampling decision.
 * @param {number} tokenId
 * @param {string} tokenText
 * @param {{ prob?: number; temperature?: number; topK?: number }} info
 * @returns {void}
 */
export function logSample(tokenId, tokenText, info) {
  if (!isEnabled('sample')) return;

  const decodeStep = getDecodeStep();
  const tag = decodeStep === 0 ? '[SAMPLE][PREFILL]' : `[SAMPLE][S${decodeStep}]`;
  let msg = `${tag} -> ${tokenId} "${tokenText}"`;
  if (info.prob !== undefined) msg += ` p=${(info.prob * 100).toFixed(1)}%`;
  if (info.temperature !== undefined) msg += ` T=${info.temperature}`;

  log.debug('Debug', msg);
}

/**
 * Log GPU buffer I/O.
 * @param {'read' | 'write' | 'copy'} op
 * @param {string} label
 * @param {number} bytes
 * @returns {void}
 */
export function logIO(op, label, bytes) {
  if (!isEnabled('io')) return;

  const tag = '[IO]';
  const kb = (bytes / 1024).toFixed(1);
  log.debug('Debug', `${tag} ${op} ${label}: ${kb}KB`);
}

/**
 * Log performance timing.
 * @param {string} phase
 * @param {number} ms
 * @param {Record<string, number | string>} [extra]
 * @returns {void}
 */
export function logPerf(phase, ms, extra) {
  if (!isEnabled('perf')) return;

  const tag = '[PERF]';
  let msg = `${tag} ${phase}: ${ms.toFixed(1)}ms`;
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      msg += ` ${k}=${typeof v === 'number' ? v.toFixed(1) : v}`;
    }
  }

  log.debug('Debug', msg);
}
