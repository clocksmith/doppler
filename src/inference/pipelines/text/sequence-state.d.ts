export interface SequenceState {
  isGenerating: boolean;
  currentSeqLen: number;
  kvCache?: { truncate?(seqLen: number): void } | null;
}

/** Rejects non-number, fractional, negative, unsafe, and increasing lengths before mutation. */
export function resetSequenceState(state: SequenceState, seqLen: number): void;
