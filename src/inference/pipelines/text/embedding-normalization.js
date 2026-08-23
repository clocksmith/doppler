import { runRMSNorm, recordRMSNorm } from '../../../gpu/kernel-selector.js';
import { getConstantVectorBuffer } from '../../../gpu/constant-buffer.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { runProbes } from './probes.js';

export async function finalizeEmbeddingOutput(tensor, normalization, options) {
  const {
    recorder = null,
    numTokens,
    hiddenSize,
    outputBuffer = null,
    probeStage = 'embed_out',
    debugProbes = null,
    operatorDiagnostics = null,
  } = options;
  let output = tensor;
  if (normalization) {
    if (outputBuffer && outputBuffer === tensor.buffer) {
      throw new Error('[Embed] weightless RMSNorm output must not alias its input buffer.');
    }
    const unitWeight = getConstantVectorBuffer(hiddenSize, 1, 'embedding_norm_ones');
    const normOptions = {
      batchSize: numTokens,
      hiddenSize,
      outputBuffer,
      rmsNormWeightOffset: false,
      label: 'embedding_norm',
    };
    output = recorder
      ? await recordRMSNorm(recorder, tensor, unitWeight, normalization.eps, normOptions)
      : await runRMSNorm(tensor, unitWeight, normalization.eps, normOptions);
    if (recorder) recorder.trackTemporaryBuffer(tensor.buffer);
    else releaseBuffer(tensor.buffer);
  }
  await runProbes(probeStage, output.buffer, {
    numTokens, hiddenSize, probes: debugProbes, recorder, operatorDiagnostics, dtype: output.dtype,
  });
  return output;
}
