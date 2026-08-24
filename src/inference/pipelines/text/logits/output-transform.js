import { releaseBuffer } from '../../../../memory/buffer-pool.js';
import { runProbes } from '../probes.js';
import { resolveLogitOutputScale } from './scale-policy.js';
import {
  recordFinalizeLogitsTensor,
  runFinalizeLogitsTensor,
} from '../../../../gpu/kernels/logit-finalize.js';

export async function finalizeLogitOutputTensor(tensor, config, options) {
  const {
    recorder = null,
    numTokens,
    vocabSize,
    targetVocabSize = vocabSize,
    bias = null,
    applySoftcap = false,
    operatorDiagnostics = null,
  } = options;
  const transform = recorder ? recordFinalizeLogitsTensor : runFinalizeLogitsTensor;
  const output = recorder
    ? await transform(recorder, tensor, {
      rowCount: numTokens,
      sourceColumns: vocabSize,
      targetColumns: targetVocabSize,
      bias,
      outputScale: resolveLogitOutputScale(config),
      softcap: applySoftcap && config.finalLogitSoftcapping != null
        ? Number(config.finalLogitSoftcapping)
        : 0,
    })
    : await transform(tensor, {
      rowCount: numTokens,
      sourceColumns: vocabSize,
      targetColumns: targetVocabSize,
      bias,
      outputScale: resolveLogitOutputScale(config),
      softcap: applySoftcap && config.finalLogitSoftcapping != null
        ? Number(config.finalLogitSoftcapping)
        : 0,
    });
  if (recorder) recorder.trackTemporaryBuffer(tensor.buffer);
  else releaseBuffer(tensor.buffer);
  await runProbes('logits', output.buffer, {
    numTokens, hiddenSize: targetVocabSize, recorder, operatorDiagnostics, dtype: output.dtype,
  });
  return output;
}
