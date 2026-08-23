import { readBuffer } from '../../../memory/buffer-pool.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { decodeReadback } from './debug-utils/index.js';
import { applyRepetitionPenalty } from './sampling.js';
import { extractLastPositionLogits, finalizeLogits } from './logits/index.js';

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
  const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsDtype });
  const logitsData = await readBuffer(logitsBuffer, vocabSize * logitsBytes);
  const rawLogits = decodeReadback(logitsData, logitsDtype);
  const finalizedLogits = await finalizeLogits(
    rawLogits,
    1,
    vocabSize,
    config.vocabSize,
    config,
    state.runtimeConfig.shared.debug.probes,
    state.operatorDiagnostics
  );
  const observedLogits = extractLastPositionLogits(finalizedLogits, 1, config.vocabSize);
  applyRepetitionPenalty(observedLogits, currentIds, opts.repetitionPenalty);
  return emitObservedLogits(opts.onLogits, observedLogits, tokenId, currentIds);
}
