import type { Tensor } from '../../../../gpu/tensor.js';
import type { KernelPathSchema } from '../../../../config/schema/index.js';
import type { LayerWeights } from '../weights.js';
export interface QwenSharedExpertOptions {
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
