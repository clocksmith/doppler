import type { Tensor } from '../../../../gpu/tensor.js';
import type { WeightBuffer } from '../../../../gpu/weight-buffer.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';

export interface QwenRouteExpertOptions {
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
}
export declare function runQwenRouteExperts(options: QwenRouteExpertOptions): Promise<Tensor>;
export declare function recordQwenRouteExperts(recorder: CommandRecorder, options: QwenRouteExpertOptions): Promise<Tensor>;
