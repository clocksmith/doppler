

import { runProbes } from '../probes.js';
import { applySoftcapping } from './cpu.js';


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

  if (config.finalLogitSoftcapping != null) {
    applySoftcapping(logits, config.finalLogitSoftcapping);
  }

  await runProbes('logits_final', logits, {
    numTokens,
    hiddenSize: vocabSize,
    probes: debugProbes,
  });

  return logits;
}
