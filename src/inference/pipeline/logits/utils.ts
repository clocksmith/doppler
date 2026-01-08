/**
 * Utility functions for logits computation.
 *
 * Provides helper functions for extracting logits and finalizing results.
 *
 * @module inference/pipeline/logits/utils
 */

import type { ProbeConfigSchema } from '../../../config/schema/index.js';
import { runProbes } from '../probes.js';
import { applySoftcapping } from './cpu.js';
import type { LogitsConfig } from './types.js';

/**
 * Extract logits for only the last position.
 *
 * Used after prefill to get logits for sampling the first generated token.
 *
 * @param logits - Full logits tensor [numTokens, vocabSize]
 * @param numTokens - Number of tokens
 * @param vocabSize - Vocabulary size
 * @returns Logits for last position [vocabSize]
 */
export function extractLastPositionLogits(
  logits: Float32Array,
  numTokens: number,
  vocabSize: number
): Float32Array {
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
 * @param rawLogits - Raw logits from matmul
 * @param numTokens - Number of tokens
 * @param matmulVocabSize - Vocab size used in matmul
 * @param vocabSize - Target vocab size
 * @param config - Logits configuration
 * @param debugProbes - Optional debug probes
 * @returns Finalized logits
 */
export async function finalizeLogits(
  rawLogits: Float32Array,
  numTokens: number,
  matmulVocabSize: number,
  vocabSize: number,
  config: LogitsConfig,
  debugProbes?: ProbeConfigSchema[] | null
): Promise<Float32Array> {
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
