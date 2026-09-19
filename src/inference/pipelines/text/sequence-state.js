// Sequence state is independent of token selection, model loading, and devices.
/** @param {import('./sequence-state.js').SequenceState} state @param {number} seqLen */
export function resetSequenceState(state, seqLen) {
  if (state.isGenerating) {
    throw new Error('InferencePipeline.resetToSeqLen: cannot reset while generation is in progress');
  }
  if (!Number.isSafeInteger(seqLen) || seqLen < 0) {
    throw new Error('InferencePipeline.resetToSeqLen: seqLen must be a finite non-negative safe integer');
  }
  if (seqLen > state.currentSeqLen) {
    throw new Error(`InferencePipeline.resetToSeqLen: target ${seqLen} exceeds currentSeqLen ${state.currentSeqLen}`);
  }
  state.kvCache?.truncate?.(seqLen);
  state.currentSeqLen = seqLen;
}
