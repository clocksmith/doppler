import { runProbes } from '../../inference/pipelines/text/probes.js';
import { applySoftcapping } from './cpu.js';
import { resolveLogitOutputScale } from '../../inference/pipelines/text/logits/scale-policy.js';

export function extractLastPositionLogits(logits, numTokens, vocabSize) {
  const lastPosLogits = new Float32Array(vocabSize);
  const lastPosOffset = (numTokens - 1) * vocabSize;

  for (let i = 0; i < vocabSize; i++) {
    lastPosLogits[i] = logits[lastPosOffset + i];
  }

  return lastPosLogits;
}

export function writeChunkLogits(
  target,
  chunk,
  numTokens,
  vocabSize,
  rowOffset,
  rowCount
) {
  for (let token = 0; token < numTokens; token += 1) {
    const sourceOffset = token * rowCount;
    const targetOffset = token * vocabSize + rowOffset;
    target.set(chunk.subarray(sourceOffset, sourceOffset + rowCount), targetOffset);
  }
}

export async function finalizeLogits(
  rawLogits,
  numTokens,
  matmulVocabSize,
  vocabSize,
  config,
  debugProbes,
  operatorDiagnostics = null
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

  const outputScale = resolveLogitOutputScale(config);
  if (outputScale !== 1) {
    for (let index = 0; index < logits.length; index += 1) logits[index] *= outputScale;
  }
  if (config.finalLogitSoftcapping != null) {
    applySoftcapping(logits, config.finalLogitSoftcapping);
  }

  await runProbes('logits_final', logits, {
    numTokens,
    hiddenSize: vocabSize,
    probes: debugProbes,
    operatorDiagnostics,
  });

  return logits;
}
