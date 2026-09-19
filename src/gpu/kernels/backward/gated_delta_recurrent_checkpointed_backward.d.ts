import type { Tensor } from '../../tensor.js';
import type { GatedDeltaRecurrentBackwardResult } from './gated_delta_recurrent_backward.js';

export interface GatedDeltaCheckpointedBackwardInputs {
  query: Tensor;
  key: Tensor;
  value: Tensor;
  logDecay: Tensor;
  beta: Tensor;
  checkpoints: Tensor;
  gradOutput: Tensor;
}

export interface GatedDeltaCheckpointedBackwardOptions {
  totalTokens: number;
  numHeads: number;
  keyDim: number;
  valueDim: number;
  checkpointInterval: number;
  queryScale: number;
}

export interface GatedDeltaCheckpointedBackwardResult extends GatedDeltaRecurrentBackwardResult {
  blockCount: number;
  checkpointInterval: number;
}

export declare function runGatedDeltaRecurrentCheckpointedBackward(
  inputs: GatedDeltaCheckpointedBackwardInputs,
  options: GatedDeltaCheckpointedBackwardOptions
): Promise<GatedDeltaCheckpointedBackwardResult>;
