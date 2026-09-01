import type { Tensor } from '../../../../gpu/tensor.js';

export interface AttentionRoPEObservationOptions {
  state: {
    ropeFreqsCos?: GPUBuffer | null;
    ropeFreqsSin?: GPUBuffer | null;
    debugProbes?: unknown[] | null;
    operatorDiagnostics?: Record<string, unknown> | null;
  };
  recorder?: Record<string, unknown> | null;
  ropeApplied: boolean;
  disableRoPE: boolean;
  qTensor: Tensor | null;
  kTensor: Tensor | null;
  layerIdx: number;
  numTokens: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
}

export declare function observeAttentionRoPE(
  options: AttentionRoPEObservationOptions
): Promise<void>;
