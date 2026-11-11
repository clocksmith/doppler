import type { PipelineState } from '../state.js';

export interface HiddenProjectionResult {
  /** Row-major logits: `rowCount` rows of `vocabSize` values. */
  logits: Float32Array;
  rowCount: number;
  vocabSize: number;
}

/**
 * Applies the model's final normalization and output head to caller-supplied
 * hidden states. Fails for hyper-connection models and paired-precision plans.
 */
export declare function projectHiddenStates(
  state: PipelineState,
  hiddenStates: Float32Array,
  rowCount: number
): Promise<HiddenProjectionResult>;
