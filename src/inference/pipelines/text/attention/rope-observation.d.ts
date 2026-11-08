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

export function observeActivityAttention(options: { state: { operatorDiagnostics?: { activity?: import('../../../../debug/activity-trace.js').ActivityTrace } | null };
  kvState: { kvLayout: string; kvStart: number; kvLenForAttention: number; startPosForMask: number };
  qTensor: Tensor; cachedKTensor: Tensor | null; cachedVTensor: Tensor | null;
  recorder: import('../../../../gpu/command-recorder.js').CommandRecorder;
  layerIdx: number; numTokens: number; numHeads: number; numKVHeads: number; headDim: number;
  attnScale: number; attnSoftcap: number; causalForAttention: boolean; effectiveSlidingWindow: number;
}): Promise<void>;
