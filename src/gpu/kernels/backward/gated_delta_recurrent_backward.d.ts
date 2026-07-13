import type { Tensor } from '../../tensor.js';

export interface GatedDeltaRecurrentBackwardInputs {
  query: Tensor;
  key: Tensor;
  value: Tensor;
  logDecay: Tensor;
  beta: Tensor;
  stateHistory: Tensor;
  gradOutput: Tensor;
}

export interface GatedDeltaRecurrentBackwardOptions {
  numTokens: number;
  numHeads: number;
  keyDim: number;
  valueDim: number;
  queryScale: number;
  gradQueryBuffer?: GPUBuffer | null;
  gradKeyBuffer?: GPUBuffer | null;
  gradValueBuffer?: GPUBuffer | null;
  gradLogDecayBuffer?: GPUBuffer | null;
  gradBetaBuffer?: GPUBuffer | null;
  gradStateBuffer?: GPUBuffer | null;
}

export interface GatedDeltaRecurrentBackwardResult {
  query: Tensor;
  key: Tensor;
  value: Tensor;
  logDecay: Tensor;
  beta: Tensor;
  initialState: Tensor;
}

export declare function runGatedDeltaRecurrentBackward(
  inputs: GatedDeltaRecurrentBackwardInputs,
  options: GatedDeltaRecurrentBackwardOptions
): Promise<GatedDeltaRecurrentBackwardResult>;
