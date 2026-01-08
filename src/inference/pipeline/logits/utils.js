/**
 * Utility functions for logits computation.
 *
 * Provides helper functions for extracting logits and finalizing results.
 *
 * @module inference/pipeline/logits/utils
 */

import { runProbes } from '../probes.js';
import { applySoftcapping } from './cpu.js';

/**
 * Extract logits for only the last position.
 *
 * Used after prefill to get logits for sampling the first generated token.
 *
 * @param {Float32Array} logits - Full logits tensor [numTokens, vocabSize]
 * @param {number} numTokens - Number of tokens
 * @param {number} vocabSize - Vocabulary size
 * @returns {Float32Array} Logits for last position [vocabSize]
 */
export function extractLastPositionLogits(
  logits,
  numTokens,
  vocabSize
) {
  const lastPosLogits = new Float32Array(vocabSize);
  const lastPosOffset = (numTokens - 1) * vocabSize;

  for (let i = 0; i < vocabSize; i++) {
    lastPosLogits[i] = logits[lastPosOffset + i];
  }

  return lastPosLogits;
}

/**
 * Finalize logits by applying padding and softcapping.
 *
 * Handles vocabulary size mismatch (padding with -Infinity)
 * and applies final logit softcapping if configured.
 *
 * @param {Float32Array} rawLogits - Raw logits from matmul
 * @param {number} numTokens - Number of tokens
 * @param {number} matmulVocabSize - Vocab size used in matmul
 * @param {number} vocabSize - Target vocab size
 * @param {import('./types.js').LogitsConfig} config - Logits configuration
 * @param {import('../../../config/schema/index.js').ProbeConfigSchema[] | null} [debugProbes] - Optional debug probes
 * @returns {Promise<Float32Array>} Finalized logits
 */
export async function finalizeLogits(
  rawLogits,
  numTokens,
  matmulVocabSize,
  vocabSize,
  config,
  debugProbes
) {
  let logits = rawLogits;

  if (matmulVocabSize < vocabSize) {
    const paddedLogits = new Float32Array(numTokens * vocabSize);
    for (let t = 0; t < numTokens; t++) {
      const srcOffset = t * matmulVocabSize;
      const dstOffset = t * vocabSize;
      for (let i = 0; i < matmulVocabSize; i++) {
        paddedLogits[dstOffset + i] = rawLogits[srcOffset + i];
      }
      for (let i = matmulVocabSize; i < vocabSize; i++) {
        paddedLogits[dstOffset + i] = -Infinity;
      }
    }
    logits = paddedLogits;
  }

  if (config.finalLogitSoftcapping) {
    applySoftcapping(logits, config.finalLogitSoftcapping);
  }

  await runProbes('logits_final', logits, {
    numTokens,
    hiddenSize: vocabSize,
    probes: debugProbes,
  });

  return logits;
}
