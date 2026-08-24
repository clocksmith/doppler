import { runScale, recordScale } from '../../../../gpu/kernel-selector.js';
import { releaseBuffer } from '../../../../memory/buffer-pool.js';
import { runProbes } from '../probes.js';
import { resolveLogitOutputScale } from './scale-policy.js';

export async function finalizeLogitOutputTensor(tensor, config, options) {
  const { recorder = null, numTokens, vocabSize, operatorDiagnostics = null } = options;
  const scale = resolveLogitOutputScale(config);
  let output = tensor;
  if (scale !== 1) {
    output = recorder
      ? await recordScale(recorder, tensor, scale, { count: numTokens * vocabSize })
      : await runScale(tensor, scale, { count: numTokens * vocabSize });
    if (recorder) recorder.trackTemporaryBuffer(tensor.buffer);
    else releaseBuffer(tensor.buffer);
  }
  await runProbes('logits', output.buffer, {
    numTokens, hiddenSize: vocabSize, recorder, operatorDiagnostics, dtype: output.dtype,
  });
  return output;
}
