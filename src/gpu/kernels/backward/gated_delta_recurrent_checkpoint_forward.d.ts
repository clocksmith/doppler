import type { Tensor } from '../../tensor.js';

export interface GatedDeltaCheckpointForwardInputs {
  query: Tensor;
  key: Tensor;
  value: Tensor;
  logDecay: Tensor;
  beta: Tensor;
  initialState: Tensor;
}

export interface GatedDeltaCheckpointForwardOptions {
  numTokens: number;
  totalTokens: number;
  tokenOffset: number;
  numHeads: number;
  keyDim: number;
  valueDim: number;
  checkpointInterval: number;
  initialStateOffsetElements: number;
  queryScale: number;
  stateBuffer?: GPUBuffer | null;
  checkpointBuffer?: GPUBuffer | null;
  outputBuffer?: GPUBuffer | null;
}

export interface GatedDeltaCheckpointForwardResult {
  output: Tensor;
  checkpoints: Tensor;
  finalState: Tensor;
  checkpointCount: number;
}

export declare function runGatedDeltaRecurrentCheckpointForward(
  inputs: GatedDeltaCheckpointForwardInputs,
  options: GatedDeltaCheckpointForwardOptions
): Promise<GatedDeltaCheckpointForwardResult>;
