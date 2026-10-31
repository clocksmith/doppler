import type { Tensor } from '../../../../gpu/tensor.js';
import type { WeightBuffer } from '../../../../gpu/weight-buffer.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';

export interface QwenRouteExpertOptions {
  /** Declared grouped projection identity, required with a production grouped path. */
  kernelPath?: import('../../../../config/schema/index.js').KernelPathSchema | null;
  phase?: 'prefill' | 'decode';
  groupedPrefill?: boolean;
  sharedSlot?: QwenSharedSlot | null;
  /** Required for a prepared single-token recorder. */
  layerIdx?: number;
  inputTensor: Tensor;
  indicesBuffer: GPUBuffer;
  weightsBuffer: GPUBuffer;
  weights: { numExperts: number; expertFormat: string; gateUp?: WeightBuffer; gate?: WeightBuffer; up?: WeightBuffer; down: WeightBuffer };
  expectedExpertFormat: string;
  numTokens: number;
  topK: number;
  hiddenSize: number;
  intermediateSize: number;
  activationDtype: string;
  hiddenActivation: string;
  swigluLimit: number | null;
}
export interface QwenSharedSlot {
  weights: import('../weights.js').LayerWeights;
  intermediateSize: number;
  kernelPath: import('../../../../config/schema/index.js').KernelPathSchema;
  phase: 'decode' | 'prefill';
  /**
   * The layer's outer residual, present exactly when the declared down variant adds it
   * (pair_combine_residual_f32); the result is then the post-residual hidden state,
   * written to `outputBuffer` when given.
   */
  residual?: { tensor: Tensor; outputBuffer: GPUBuffer | null } | null;
  /** Shared-expert gate [numTokens, 1] already computed by a fused norm + router dispatch. */
  gate?: Tensor | null;
}
export declare function runQwenRouteExperts(options: QwenRouteExpertOptions): Promise<Tensor>;
export declare function recordQwenRouteExperts(recorder: CommandRecorder, options: QwenRouteExpertOptions): Promise<Tensor>;
