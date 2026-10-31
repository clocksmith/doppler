import type { Tensor } from '../../../../gpu/tensor.js';
import type { KernelPathSchema } from '../../../../config/schema/index.js';
import type { LayerWeights } from '../weights.js';
export interface QwenSharedExpertOptions {
  /** Required whenever an explicit kernel path is supplied. */
  phase?: 'decode' | 'prefill';
  /** Required for a prepared single-token recorder or explicit kernel path. */
  layerIdx?: number;
  recorder?: import('../../../../gpu/command-recorder.js').CommandRecorder | null;
  numTokens: number;
  hiddenSize: number;
  intermediateSize: number;
  kernelPath: KernelPathSchema | null;
  swigluLimit: number | null;
}
export type QwenSharedExpertWeights = { [K in 'sharedExpertGate' | 'sharedExpertUp' | 'sharedExpertDown' | 'sharedExpertRouter']: Exclude<NonNullable<LayerWeights[K]>, Float32Array> };
export declare function validateQwenSharedExpert(input: Tensor, routed: Tensor,
  weights: LayerWeights, options: QwenSharedExpertOptions): asserts weights is QwenSharedExpertWeights;
export declare function runQwenSharedExpert(input: Tensor, routed: Tensor,
  weights: LayerWeights, options: QwenSharedExpertOptions): Promise<Tensor>;

export declare function resolveQwenSharedExpertCombine(options: QwenSharedExpertOptions): boolean;
export declare function resolveQwenSharedSlot(options: {
  kernelPath: KernelPathSchema | null;
  phase: 'decode' | 'prefill';
  layerIdx: number;
}): boolean;
