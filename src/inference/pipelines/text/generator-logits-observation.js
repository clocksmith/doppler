import { readBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../../gpu/tensor.js';
import { runFinalizeLogitsTensor } from '../../../gpu/kernels/logit-finalize.js';
import { applyRepetitionPenalty } from './sampling.js';

export function emitObservedLogits(onLogits, logits, tokenId, currentIds) {
  if (typeof onLogits !== 'function') return false;
  onLogits(logits, {
    tokenId,
    inputTokenCount: Array.isArray(currentIds) ? currentIds.length : null,
  });
  return true;
}

export async function captureObservedFusedDecodeLogits(
  state,
  opts,
  logitsBuffer,
  vocabSize,
  logitsDtype,
  tokenId,
  currentIds
) {
  if (typeof opts.onLogits !== 'function') return false;
  const config = state.modelConfig;
  const finalized = await runFinalizeLogitsTensor(
    createTensor(logitsBuffer, logitsDtype, [1, vocabSize], 'observed_decode_logits'),
    {
      rowCount: 1,
      sourceColumns: vocabSize,
      targetColumns: config.vocabSize,
      bias: null,
      outputScale: 1,
      softcap: config.finalLogitSoftcapping == null ? 0 : Number(config.finalLogitSoftcapping),
    }
  );
  const logitsData = await readBuffer(
    finalized.buffer,
    config.vocabSize * Float32Array.BYTES_PER_ELEMENT
  );
  releaseBuffer(finalized.buffer);
  const observedLogits = new Float32Array(logitsData);
  applyRepetitionPenalty(observedLogits, currentIds, opts.repetitionPenalty);
  return emitObservedLogits(opts.onLogits, observedLogits, tokenId, currentIds);
}
